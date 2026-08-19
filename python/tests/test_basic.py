"""Essential smoke tests for the hypercube_etalon wheel.

Kept lean so cibuildwheel stays short. Prove the compiled ``_core`` loads and
the map pipeline (collect → train → predict) produces sane results on the
target platform — not exhaustive façade coverage.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import hypercube_etalon
from hypercube_etalon import Etalon


def _make_patterns(dim: int, n_per_class: int, n_classes: int, seed: int = 0):
    """Simple multi-class length-N fields (tone-ish + noise)."""
    rng = np.random.default_rng(seed)
    n = 1 << dim
    fields = []
    labels = []
    for c in range(n_classes):
        for rep in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
            x = np.sin((c + 1) * t + 0.1 * rep).astype(np.float32)
            x += 0.05 * rng.standard_normal(n).astype(np.float32)
            # Class-dependent peak in the high half
            x[n // 2 + c * 2] += 1.5
            fields.append(x)
            labels.append(c)
    return np.stack(fields, axis=0), np.asarray(labels, dtype=np.int32)


@pytest.fixture(scope="module")
def cls_data():
    return _make_patterns(dim=5, n_per_class=24, n_classes=3, seed=1)


@pytest.fixture(scope="module")
def trained_cls(cls_data):
    fields, labels = cls_data
    # Gains follow the C++ demo recipes: the header defaults (0.02 input,
    # 0.02 weight) are Cascade-feeding values and barely drive tanh here.
    et = Etalon(
        dim=5,
        exciter_subcube_dim=4,
        exciter_seed=1,
        exciter_input_scaling=1.0,
        exciter_weight_scaling=0.5,
        readout_num_outputs=3,
        readout_task="classification",
        readout_num_layers=1,
        readout_conv_channels=4,
        readout_use_pooling=False,
        readout_activation="none",
        readout_lr_max=0.003,
        readout_epochs=60,
        readout_batch_size=16,
        readout_num_threads=1,
        readout_restore_best_epoch=False,
        collect_threads=1,
    )
    et.fit(fields, labels)
    return et, fields, labels


# ── Construction ──

class TestVersion:
    def test_version_string(self):
        v = hypercube_etalon.__version__
        assert isinstance(v, str) and len(v) > 0
        assert v[0].isdigit()
        # Same string as the extension module (CMake baked from _version.py)
        from hypercube_etalon import _core
        assert _core.__version__ == v


class TestConstruction:
    @pytest.mark.parametrize("dim", [4, 7])
    def test_construct(self, dim):
        et = Etalon(dim=dim, exciter_subcube_dim=3)
        assert et.dim == dim
        assert et.N == 2**dim
        assert et.num_collected == 0
        assert et.subcube_dim == 3
        assert et.walk_size == 8
        assert et.feature_size == et.N
        assert et.bypass_exciter is False

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be"):
            Etalon(dim=3, exciter_subcube_dim=3)
        with pytest.raises(ValueError, match="dim must be"):
            Etalon(dim=13, exciter_subcube_dim=4)

    def test_invalid_subcube_dim(self):
        # C++ default subcube_dim=6 is illegal at dim 5; wrapper catches early.
        with pytest.raises(ValueError, match="exciter_subcube_dim"):
            Etalon(dim=5)
        with pytest.raises(ValueError, match="exciter_subcube_dim"):
            Etalon(dim=7, exciter_subcube_dim=8)

    def test_defaults(self):
        et = Etalon(dim=6)
        assert et.readout_task == "regression"
        assert et.num_outputs == 1
        assert et.subcube_dim == 6
        assert et.bypass_exciter is False
        assert et.collect_threads == 0

    def test_repr(self):
        et = Etalon(dim=5, exciter_subcube_dim=4)
        r = repr(et)
        assert "dim=5" in r
        assert "N=32" in r


# ── Classification pipeline ──

class TestClassification:
    def test_fit_train_acc(self, trained_cls):
        et, _, _ = trained_cls
        acc = et.accuracy_on_collected()
        assert acc > 0.85, f"train accuracy too low: {acc}"

    def test_predict_class_shape(self, trained_cls):
        et, fields, labels = trained_cls
        pred = et.predict_class(fields[0])
        assert isinstance(pred, int)
        assert 0 <= pred < et.num_outputs
        logits = et.predict(fields[0])
        assert logits.shape == (et.num_outputs,)
        assert logits.dtype == np.float32
        assert int(np.argmax(logits)) == pred

    def test_heldout_sane(self, trained_cls):
        et, _, _ = trained_cls
        # Fresh draws with different seed — should still beat chance
        fields_te, labels_te = _make_patterns(5, 16, 3, seed=99)
        acc = et.accuracy(fields_te, labels_te)
        assert acc > 0.5, f"test accuracy too low: {acc}"
        # Bulk accuracy agrees with per-sample predict_class
        correct = sum(
            et.predict_class(fields_te[i]) == int(labels_te[i])
            for i in range(len(labels_te))
        )
        assert acc == pytest.approx(correct / len(labels_te))


# ── Regression ──

class TestRegression:
    def test_r2_on_collected(self):
        dim = 5
        n = 1 << dim
        rng = np.random.default_rng(3)
        fields = rng.standard_normal((80, n), dtype=np.float32)
        # Strong scalar signal in the field so a short train can move R²
        targets = (2.0 * fields[:, 0:1] + 0.05 * rng.standard_normal((80, 1))).astype(
            np.float32
        )
        et = Etalon(
            dim=dim,
            exciter_subcube_dim=4,
            exciter_input_scaling=1.0,
            exciter_weight_scaling=0.5,
            readout_num_outputs=1,
            readout_task="regression",
            readout_epochs=80,
            readout_num_threads=1,
            collect_threads=1,
            readout_restore_best_epoch=False,
        )
        et.fit(fields, targets)
        r2 = et.r2_on_collected()
        assert np.isfinite(r2), f"R² not finite: {r2}"
        # Smoke: should beat "always predict mean" by a bit on this easy signal
        assert r2 > 0.0, f"R² too low: {r2}"
        y = et.predict(fields[0])
        assert y.shape == (1,)
        assert y.dtype == np.float32
        # Held-out R² is finite on fresh draws
        fields_te = rng.standard_normal((20, n), dtype=np.float32)
        targets_te = (2.0 * fields_te[:, 0:1]).astype(np.float32)
        assert np.isfinite(et.r2(fields_te, targets_te))


# ── Map helpers ──

class TestMap:
    def test_run_last_features(self):
        et = Etalon(dim=5, exciter_subcube_dim=4,
                    exciter_input_scaling=1.0, exciter_weight_scaling=0.5)
        x = np.zeros(et.N, dtype=np.float32)
        x[0] = 1.0
        et.run(x)
        feat = et.last_features()
        assert feat.shape == (et.feature_size,)
        assert feat.dtype == np.float32
        assert np.all(np.isfinite(feat))
        # The transit is a real map — features are not just the input copy.
        assert not np.allclose(feat, x)

    def test_bypass_exciter_copies_field(self):
        et = Etalon(dim=5, exciter_subcube_dim=4, bypass_exciter=True)
        assert et.bypass_exciter is True
        x = np.linspace(-1, 1, et.N, dtype=np.float32)
        et.run(x)
        np.testing.assert_allclose(et.last_features(), x, atol=1e-7)

    def test_deterministic_map(self):
        et = Etalon(dim=5, exciter_subcube_dim=4,
                    exciter_input_scaling=1.0, exciter_weight_scaling=0.5)
        x = np.linspace(-1, 1, et.N, dtype=np.float32)
        et.run(x)
        a = et.last_features().copy()
        et.run(x)
        b = et.last_features().copy()
        np.testing.assert_array_equal(a, b)

    def test_field_size_check(self):
        et = Etalon(dim=5, exciter_subcube_dim=4)
        with pytest.raises(Exception, match="must equal N"):
            et.run(np.zeros(16, dtype=np.float32))


# ── Serial collect ──

class TestSerialCollect:
    def test_collect_appends(self, cls_data):
        fields, labels = cls_data
        et = Etalon(
            dim=5,
            exciter_subcube_dim=4,
            readout_num_outputs=3,
            readout_task="classification",
            collect_threads=1,
        )
        assert et.num_collected == 0
        et.collect(fields[0], int(labels[0]))
        et.collect(fields[1], int(labels[1]))
        assert et.num_collected == 2
        # Serial collect leaves this sample's features in last_features
        assert et.last_features().shape == (et.N,)
        et.clear_collected()
        assert et.num_collected == 0

    def test_collect_rejects_bool_label(self, cls_data):
        fields, _ = cls_data
        et = Etalon(
            dim=5,
            exciter_subcube_dim=4,
            readout_num_outputs=3,
            readout_task="classification",
        )
        with pytest.raises(TypeError, match="integer"):
            et.collect(fields[0], True)

    def test_label_out_of_range(self, cls_data):
        fields, _ = cls_data
        et = Etalon(
            dim=5,
            exciter_subcube_dim=4,
            readout_num_outputs=3,
            readout_task="classification",
        )
        with pytest.raises(ValueError):
            et.collect(fields[0], 3)


# ── Persistence ──

class TestPersistence:
    def test_pickle_roundtrip(self, trained_cls):
        et, fields, labels = trained_cls
        acc_before = et.accuracy_on_collected()
        loaded = pickle.loads(pickle.dumps(et))
        assert loaded.num_collected == 0
        assert loaded.dim == et.dim
        assert loaded.subcube_dim == et.subcube_dim
        # Same weights → same predictions
        for i in range(8):
            assert loaded.predict_class(fields[i]) == et.predict_class(fields[i])
        # Retrain not required for infer
        assert acc_before > 0.85

    def test_save_load(self, trained_cls, tmp_path):
        et, fields, _ = trained_cls
        path = tmp_path / "model.pkl"
        et.save(path)
        loaded = Etalon.load(path)
        assert loaded.predict_class(fields[0]) == et.predict_class(fields[0])

    def test_hcnn_model_roundtrip(self, trained_cls, tmp_path):
        et, fields, _ = trained_cls
        stem = tmp_path / "export" / "stem"
        (tmp_path / "export").mkdir()
        et.save_readout_hcnn_model(stem)
        assert (tmp_path / "export" / "stem.hcnw").is_file()
        assert (tmp_path / "export" / "stem.arch.json").is_file()
        # Fresh instance, same architecture knobs → identical predictions
        fresh = Etalon(
            dim=5,
            exciter_subcube_dim=4,
            exciter_seed=1,
            exciter_input_scaling=1.0,
            exciter_weight_scaling=0.5,
            readout_num_outputs=3,
            readout_task="classification",
            readout_num_layers=1,
            readout_conv_channels=4,
            readout_use_pooling=False,
            readout_activation="none",
            readout_num_threads=1,
        )
        fresh.load_readout_hcnn_model(stem)
        for i in range(8):
            assert fresh.predict_class(fields[i]) == et.predict_class(fields[i])
        summary = et.readout_arch_summary()
        assert isinstance(summary, str) and "weight_count" in summary


# ── Surface ──

class TestSurface:
    EXPECTED = [
        "run",
        "last_features",
        "clear_collected",
        "collect",
        "collect_batch",
        "fit",
        "train",
        "predict",
        "predict_class",
        "accuracy_on_collected",
        "r2_on_collected",
        "accuracy",
        "r2",
        "save",
        "load",
        "save_readout_hcnn_model",
        "load_readout_hcnn_model",
        "readout_arch_summary",
    ]

    @pytest.mark.parametrize("name", EXPECTED)
    def test_method_present(self, name):
        assert callable(getattr(Etalon, name, None)), f"Etalon.{name} missing"

# HypercubeEtalon Python SDK

Static fields have no natural clock. HypercubeEtalon preprocesses each
length-N field with **one frozen hypercube stage** — an etalon transit — and
trains a small
[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) readout on the
**transit output**. One class — `Etalon` — owns collect → train → predict.

This is a **map API**, not a stream API. There is no per-tick input sequence
and no next-step `fit` on a 1D signal (that is
[HypercubeESN](https://github.com/dliptak001/HypercubeESN)). Each sample is
one full field, mapped once.

C++ core and contracts: **[CPP_SDK.md](CPP_SDK.md)**.  
PyPI-facing package story: **[python/README.md](../python/README.md)**.  
Package version: single source `python/hypercube_etalon/_version.py`
(`hypercube_etalon.__version__` and wheel metadata both read it).

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [What a map is](#what-a-map-is)
- [Pipeline vocabulary](#pipeline-vocabulary)
- [API reference](#api-reference)
- [Input data layout](#input-data-layout)
- [Data types](#data-types)
- [Error handling](#error-handling)
- [Model persistence](#model-persistence)
- [Limitations](#limitations)
- [Dependencies](#dependencies)

## Installation

### From PyPI (preferred)

Pre-built **wheels** — no compiler required:

```bash
pip install hypercube-etalon
```

Import as `import hypercube_etalon as he` (PyPI name `hypercube-etalon`).
Wheels cover Python 3.10–3.14 on common Windows (x64), Linux (x86_64,
aarch64), and macOS (x86_64, arm64) builds. NumPy is the only runtime
dependency.

### From source (full repository)

Compile only from a **full clone** of HypercubeEtalon. The extension links
the C++ core and vendored HypercubeCNN that sit **outside** the `python/`
package directory; a `python/`-only tree is not enough.

Requirements: Python 3.10+, C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+),
CMake 3.20+, scikit-build-core, pybind11, NumPy.

```bash
git clone https://github.com/dliptak001/HypercubeEtalon.git
cd HypercubeEtalon/python
pip install .
```

On Windows with MinGW (e.g. CLion toolchain):

```powershell
pip install scikit-build-core pybind11 numpy
$env:PATH = "C:\path\to\mingw\bin;" + $env:PATH
$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_MAKE_PROGRAM = "C:\path\to\ninja.exe"
$env:CC = "C:\path\to\mingw\bin\gcc.exe"
$env:CXX = "C:\path\to\mingw\bin\g++.exe"
pip install . --no-build-isolation
```

### Running tests

From the `python/` directory after install:

```bash
pip install ".[test]"
pytest tests/ -v --import-mode=importlib
```

Or from the repository root: `pytest python/tests/ -v --import-mode=importlib`.
Importlib mode avoids the source tree shadowing the installed `_core`
extension. Use the `pytest` entry point, not `python -m pytest` — the latter
puts the current directory on `sys.path`, and from `python/` the source
package (which has no compiled `_core`) then shadows the installed one.

### Examples

The [Quick start](#quick-start) below is enough after `pip install`. Longer
demos live in the **git tree** under
[`python/examples/`](../python/examples/README.md) — they are **not** part of
the wheel. From a clone, repository root:

```bash
pip install hypercube-etalon   # or: pip install ./python
python python/examples/synthetic_classification.py
```

## Quick start

```python
import numpy as np
import hypercube_etalon as he

dim = 7
N = 1 << dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((128, N), dtype=np.float32)
labels = rng.integers(0, 4, size=128, dtype=np.int32)

et = he.Etalon(
    dim=dim,
    exciter_subcube_dim=5,
    exciter_input_scaling=1.0,
    exciter_weight_scaling=0.15,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
et.fit(fields, labels)

print(et.accuracy_on_collected())  # train-set only — not a test score
print(et.predict_class(fields[0]))
```

### Explicit (full control)

```python
et = he.Etalon(
    dim=6,
    exciter_subcube_dim=5,
    exciter_input_scaling=1.0,
    exciter_weight_scaling=0.15,
    readout_num_outputs=3,
    readout_task="classification",
)
et.collect_batch(fields_train, labels_train)
et.train()
logits = et.predict(fields_test[0])   # shape (num_outputs,)
cls = et.predict_class(fields_test[0])
test_acc = et.accuracy(fields_test, labels_test)  # held-out, fresh maps
```

`fit` is `clear_collected` (optional) → `collect_batch` → `train`. Prefer
`fit` for a first pass; use collect/train when you append batches or retrain
without re-mapping every field.

## What a map is

```text
x  (length-N field, host-packed)
    │
    ▼
 etalon transit  (or a plain copy, when bypass_exciter)
    │
    ▼
 features (N)  →  HypercubeCNN  →  logits / values
```

- **N = 2^dim** vertices / field length (dim 4…12; prefer ≥ 5 so a pooled
  readout has room).
- Exciter weights are **frozen** after construction; only the readout trains.
- **Predict** always runs a **fresh** map.
- Host packing (MNIST → N, spectra → N, …) is **your** problem — this package
  does not reshape domain data onto the cube.

The CNN head never sees the original field; it sees what the wave leaves
behind. (Unless `bypass_exciter=True` — the built-in ablation where the head
sees a copy of the field.)

## Pipeline vocabulary

| Term | Meaning |
|------|---------|
| **Field** | Length-N float32 vector on the cube (you pack domain data) |
| **Transit** | One frozen etalon sweep: field in, same-length field out |
| **Map** | Transit (or copy, under bypass) → features |
| **Collect** | Run a map → append features + label/target |
| **Train** | Batch-train HCNN on all collected samples |
| **Predict** | Fresh map + readout forward |
| **N** | Vertices / field length = 2^dim |
| **subcube_dim** | Etalon face size; one walk covers 2^subcube_dim vertices |
| **bypass_exciter** | Skip the transit; features = a copy of the field (ablation) |

Unlike HypercubeWTF there is no orbit: no `T`, no `readout_slices` (B), no
reservoir knobs. Features are always the transit output (length N).

## API reference

### Constructor `Etalon(dim, **kwargs)`

All knobs are fixed at construction (same contract as C++ `EtalonConfig`).
`dim` goes to the Exciter; the readout auto-sizes to match.

```python
import hypercube_etalon as he

et = he.Etalon(
    dim=7,                          # required; 4–12; N = 2^dim
    bypass_exciter=False,           # True = ablation (features = the field)
    collect_threads=0,              # 0 = auto
    exciter_seed=7934791766227647176,
    exciter_input_scaling=0.02,     # demos run ~1.0 — see gain note below
    exciter_weight_scaling=0.02,    # demos run 0.15–0.5
    exciter_subcube_dim=6,          # [1, dim] — set <= dim when dim < 6!
    readout_num_outputs=1,
    readout_task="regression",      # or "classification"
    # … readout_* kwargs below
)
```

#### Etalon and exciter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | required | Hypercube dimension **[4, 12]**; prefer ≥ 5 for pooled readouts. N = 2^dim. |
| `bypass_exciter` | `bool` | `False` | Skip the transit; the readout sees a copy of the field (ablation path). |
| `collect_threads` | `int` | `0` | Bulk workers: 0 = auto, 1 = serial, K = K workers. |
| `exciter_seed` | `int` | `7934791766227647176` | Exciter weight-init seed (matches C++). |
| `exciter_input_scaling` | `float` | `0.02` | Scalar applied once to the field before the transit. |
| `exciter_weight_scaling` | `float` | `0.02` | Exciter neighbor weights are U(-1, 1) × this. |
| `exciter_subcube_dim` | `int` | `6` | Etalon face size; walk covers 2^subcube_dim vertices. Valid **[1, dim]** — the default 6 is rejected below dim 6. |

**Gain tuning:** the header defaults (`exciter_input_scaling=0.02`,
`exciter_weight_scaling=0.02`) drive the tanh sites very weakly — on many
tasks the features come out crushed toward zero and the readout cannot learn.
The in-tree demos run `input_scaling` ≈ 1.0 and `weight_scaling` 0.15–0.5.
Probe with `run(x)` + `last_features()` until the output is alive.

#### Readout (HCNN)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `readout_num_outputs` | `int` | `1` | Classes (classification) or regression width. |
| `readout_task` | `str` | `"regression"` | `"regression"` or `"classification"`. |
| `readout_num_layers` | `int` | `1` | Conv(+Pool) stages. **`0` = auto** `min(dim−2, 2)`. |
| `readout_conv_channels` | `int` | `16` | Base channel count for the first conv. |
| `readout_epochs` | `int` | `200` | Batch-train epochs. |
| `readout_batch_size` | `int` | `32` | Mini-batch size. |
| `readout_lr_max` | `float` | `0.0015` | Cosine peak LR. Keep ≤ ~0.005 to avoid NaN. |
| `readout_lr_min_frac` | `float` | `0.01` | Floor = `lr_max * lr_min_frac`. |
| `readout_lr_decay_epochs` | `int` | `0` | Cosine horizon; `0` = use `readout_epochs`. |
| `readout_weight_decay` | `float` | `0.0` | L2 on CNN weights. |
| `readout_momentum` | `float` | `0.9` | SGD momentum; ignored under the default Adam optimizer. |
| `readout_activation` | `str` | `"tanh"` | `"tanh"`, `"relu"`, `"leaky_relu"`, or `"none"`. |
| `readout_seed` | `int` | `42` | CNN weight-init seed. |
| `readout_num_threads` | `int` | `0` | HCNN workers: 0 = auto, 1 = single-threaded. |
| `readout_restore_best_epoch` | `bool` | `True` | Restore best-epoch weights after batch train. |
| `readout_best_epoch_holdout_frac` | `float` | `0.0` | Tail hold-out for best-epoch scoring; 0 = full train set. |
| `readout_use_pooling` | `bool` | `True` | Antipodal pool after each conv. |

**Not bound in Python yet** (C++ `ReadoutConfig` only): optimizer choice (C++
default Adam), pool type, channel growth, batch-norm. C++ defaults apply.

### Methods

| Method | Role |
|--------|------|
| `run(x)` | Map one field (no training-set append). Updates `last_features()`. |
| `last_features()` | Length-N float32 from the most recent completed map — updated by **every** map, including bulk calls (last row). |
| `clear_collected()` | Drop the batch training buffer. |
| `collect(x, target)` | Serial append one sample (label or regression vector). |
| `collect_batch(fields, targets)` | Bulk parallel append. |
| `fit(fields, targets, *, clear=True)` | Optional clear → collect → train. Returns `self`. |
| `train()` | Batch-train HCNN on all collected samples. Does not clear the set. |
| `predict(x)` | Fresh map + forward → shape `(num_outputs,)` float32. |
| `predict_class(x)` | Fresh map + argmax class (classification task only). |
| `accuracy_on_collected()` | Accuracy on the **collected training set** only. |
| `r2_on_collected()` | R² on the **collected training set** only. |
| `accuracy(fields, labels)` | Fresh bulk maps + accuracy on a **held-out** set (classification). |
| `r2(fields, targets)` | Fresh bulk maps + R² on a **held-out** set (regression). |
| `save(path)` / `load(path)` | Pickle constructor config + readout weights. |
| `save_readout_hcnn_model(path_stem)` | Portable `stem.hcnw` + `stem.arch.json`. |
| `load_readout_hcnn_model(path_stem, *, mode="eval")` | Load HCNW into this instance (`"eval"` or `"resume_train"`). |
| `readout_arch_summary()` | Human-readable HCNN architecture and parameter counts. |

### Properties

| Property | Meaning |
|----------|---------|
| `dim`, `N` | Geometry (N = 2^dim) |
| `subcube_dim`, `walk_size` | Etalon face dim and 2^subcube_dim |
| `bypass_exciter` | True when the transit is skipped (ablation) |
| `feature_size` | Floats per sample / `last_features` — always N |
| `num_collected` | Samples in the batch training buffer |
| `num_outputs` | Readout width |
| `exciter_seed` | Exciter weight seed |
| `exciter_input_scaling`, `exciter_weight_scaling` | Exciter config mirrors |
| `collect_threads` | Bulk-worker preference (0 = auto) |
| `readout_task` | `"regression"` or `"classification"` |
| `readout_best_epoch` | 1-based best epoch after restore; else 0 |

## Input data layout

- **Fields** must be length **N** per sample. Prefer shape `(count, N)` for
  bulk APIs; a flat length `count * N` vector is also accepted.
- **Host packing** (images, spectra, sensors → N) is outside this package.
- **Classification labels**: integer class indices in **`[0, num_outputs)`**
  (enforced at collect / scoring). Shape `(count,)` for bulk calls.
- **Regression targets**: shape `(count, num_outputs)` float32 (or flat
  `count * num_outputs`).
- Single-sample methods accept any array that ravel-flattens to the right
  length.

## Data types

| Role | Preferred type | Notes |
|------|----------------|-------|
| Fields / features / predictions | `float32` | Other dtypes converted via NumPy to contiguous float32 |
| Class labels | `int32` (or Python `int`) | Must be in `[0, num_outputs)` (C++ enforces) |
| Bool as a class label | rejected on serial collect | `collect` raises `TypeError`; use an integer index. Bulk `collect_batch` coerces via int32 (do not rely on bool labels). |

## Error handling

Python-side checks raise `ValueError` or `TypeError` with a short message (bad
`dim`, `exciter_subcube_dim`, task string, activation, field shape, label
count, …). Native `std::invalid_argument` maps to `ValueError`; other C++
failures typically surface as `RuntimeError` via pybind11.

Typical mistakes:

- Field length ≠ N
- `exciter_subcube_dim` left at its default 6 with dim 4 or 5
- Bulk `fields` / `targets` row counts disagree
- Class label outside `[0, num_outputs)`
- `predict_class` / `accuracy` on a regression model
- Calling `train` or `accuracy_on_collected` with an empty collected set

## Model persistence

| Mechanism | What is stored | Collected samples? |
|-----------|----------------|---------------------|
| `save` / `pickle` | Constructor config + readout weight blob | **No** (`num_collected` is 0 after load) |
| `save_readout_hcnn_model` | Portable HCNW + arch sidecar | **No** |

Pickle version is bumped when the serialized layout changes; newer libraries
reject unknown future versions with an upgrade message.

```python
et.save("model.pkl")
et2 = he.Etalon.load("model.pkl")  # same ctor knobs + weights; empty collect buffer

et.save_readout_hcnn_model("export/stem")   # stem.hcnw + stem.arch.json
# Target instance must build a matching HCNN input shape / task (same dim and
# readout_* architecture knobs as the exporter — not only dim/outputs).
et3 = he.Etalon(
    dim=et.dim,
    exciter_subcube_dim=et.subcube_dim,
    readout_num_outputs=et.num_outputs,
    readout_task=et.readout_task,
    # plus any non-default readout_num_layers / channels / pooling / …
)
et3.load_readout_hcnn_model("export/stem", mode="eval")
```

The preprocessor itself is never serialized — it reconstructs exactly from
the constructor seed and scalars. A pickle therefore captures the whole
product: config in, identical frozen transit out, plus the trained readout.

Prefer `save` / `load` when you want a full Python round-trip of the product
config. Prefer HCNW when you need a portable HypercubeCNN weight export.

**Security:** `load` uses `pickle.load`. Never load untrusted files.

## Limitations

- One `Etalon` instance is **not thread-safe** for concurrent public calls
  from multiple host threads. Bulk parallelism is internal only.
- `accuracy_on_collected` / `r2_on_collected` only score samples you already
  collected (and typically trained on). Use `accuracy` / `r2` on held-out
  fields for real evaluation.
- No train-noise knob (HypercubeWTF has one); add noise in host code. The
  bypass ablation, by contrast, is built in (`bypass_exciter=True`).
- A few readout knobs remain C++-only (optimizer, pool type, channel growth,
  batch-norm); see constructor tables above.
- Native contracts, map mechanics, and host integration detail:
  **[CPP_SDK.md](CPP_SDK.md)**.

## Dependencies

| Layer | What |
|-------|------|
| Runtime | NumPy |
| Wheel install | No compiler |
| From-source build | Full repo clone, C++23, CMake ≥ 3.20, scikit-build-core, pybind11 |

The HypercubeCNN readout is built into the extension — no separate HCNN
package.

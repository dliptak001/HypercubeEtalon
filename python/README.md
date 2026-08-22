# Hypercube Etalon

This package is the **Python** surface for HypercubeEtalon
(`import hypercube_etalon`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**.

HypercubeEtalon processes spatial data of the kind presented to a CNN.
It is built from three core classes.

The **Etalon** class wraps the other two and manages training and
prediction.

The other two form a pipeline: preprocessor → readout.

The **Exciter** class is a preprocessing stage that consumes input
patterns, mixes them nonlinearly, and returns a field with the same
dimensions as the input.

The **Readout** class is a small HypercubeCNN that classifies or
regresses that field.

This is not reservoir computing.

The point of this experiment is to see if a preprocessing stage in
front of HypercubeCNN outperforms HypercubeCNN by itself. HypercubeWTF
has the same goal; it just does it a slightly different way, using a
**reservoir** with synthetic time, whereas here the preprocessor is an
**etalon**. The aim is a hypercube preprocessor effective enough that
the readout can be a single layer with a single convolutional channel
(weird, I know) and no pooling. Then training is fast, the memory
footprint is small, and little to no architectural engineering is
required for the CNN.

---

<p align="center">
  <strong>HypercubeAI ecosystem</strong><br/>
</p>

<p align="center">
  <a href="https://github.com/dliptak001/HypercubeESN"><strong>HypercubeESN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCNN"><strong>HypercubeCNN</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeHopfield"><strong>HypercubeHopfield</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeWTF"><strong>HypercubeWTF</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeEtalon"><strong>HypercubeEtalon</strong></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/dliptak001/HypercubeCascade"><strong>HypercubeCascade</strong></a>
</p>

HypercubeEtalon is an experiment in the **HypercubeAI** project — our quest to
systematically re-implement classical neural architectures on a Boolean
hypercube topology instead of Euclidean grids or random graphs. The central
thesis is “topology-native intelligence”: the hypercube’s algebraic structure
(vertex-transitive symmetry, Hamming geometry, bitwise addressing) can serve
as a first-class computational substrate.

- **A topology you don’t store** — the graph is specified: connectivity is
  implicit in the vertex indices; with a seed and a few config scalars the whole
  reservoir reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the reservoir’s output with
  zero geometric distortion, and the learned kernels exploit the same locality
  that generated the dynamics. The data never leaves the hypercube it was born
  on.

Each product in the family is a different architecture on that same foundation.

---

## The Etalon

I now suspect that the hypercube will someday be recognized as the most
natural (least contrived) and at the same time the most powerful neural
network substrate that can possibly be realized.

The **Etalon** construct is just another example of how incredibly
elegant solutions can be built on that substrate.

Etalon is a term borrowed from optics. The physical etalon is a pair of
plane-parallel, highly reflective surfaces (mirrors) between which an
optical signal propagates. It is used for laser resonators,
interferometric measurement, and filtering.

On the hypercube, an `etalon` is a pair of vertices: any vertex and its
antipode. A hypercube has `N` vertices and therefore `N` etalons on the
full cube. There are far more than `N` once the cube is subdivided into
subcubes, each of which carries its own set of etalons.

The HypercubeEtalon design treats a vertex and its antipode as a
reflective cavity. All vertices in between contribute to the evolution
of an input signal. That procedure is an **etalon transit**. It goes
something like this.

    LOOP:

        Pick a vertex r and its antipode r'. This defines an etalon.

        Copy the input field onto the cube. That overlay is the initial
        condition, and it is the same for every etalon.

        Starting at r, form the weighted sum of its nearest neighbors
        and write tanh of that sum into r.

        Move to the next vertex along the etalon, form the neighbor
        sum, and write tanh of that sum into that vertex.

        Order is causal: a later vertex sees values the earlier ones
        just wrote.

        Continue until the transit reaches the antipode r', then turn
        around and go back to the starting vertex.

        The starting vertex is then updated for the second time. That
        is its final value, which is copied to an output buffer.

        For that etalon the task is done.

    GOTO LOOP

The loop repeats until every vertex (every etalon) has been processed,
which fully populates the output buffer.

---

## White noise filter

The etalon preprocessor behaves as a near unity passthrough at low
to no white noise levels, and offers a meaningful filtering effect
at moderate to high noise levels. The write-up is
[`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/mnist/WhiteNoiseFilter.md).

![MNIST test noise: etalon transit vs Bypass](https://raw.githubusercontent.com/dliptak001/HypercubeEtalon/main/examples/mnist/etalon_mnist_noise_comp.png)

## Raman baseline extraction (a vibrational spectroscopy application)

The first real-world test is Raman spectra: recover the slow
fluorescence background under sharp molecular peaks without
lifting the baseline into the bands or cutting trenches beneath
them. Polynomials, asymmetric least squares, and ordinary
convolutional nets tend to follow the empty stretches well and then
fail where it matters, under peaks and peak clusters. Analysts have
worked around that for decades with spectrum-specific cleanup,
because no method identifies and extracts a true baseline across a
broad range of peak intensities and baseline characteristics
without occasional, and often frequent, human intervention.

The Etalon appears to have solved that problem (albeit on synthetic
data only so far).

Trained for 60 epochs on the LCOHard set — 10,000 synthetic LiCoO₂
(lithium cobalt oxide) spectra — it scores a validation RMSE of
4.77 counts on 2,000 held-out spectra whose baselines span
hundreds of counts.

Below are four held-out validation spectra: grey is the raw
spectrum, red the true baseline, blue the extract. For all four
shown here, and for each of the remaining 1996 validation spectra
not shown, baseline identification is, **WITHOUT EXCEPTION**,
quite remarkable.

And it does this with the thin readout the project aims for: one
HypercubeCNN layer, one convolutional channel, no pooling.

In our judgment this at least matches the best of the established
techniques on spectra like these, and very likely beats them.

![Held-out validation extract, spectra 581 through 584](https://raw.githubusercontent.com/dliptak001/HypercubeEtalon/main/examples/RamanBaselineExtraction/extracted_baselines_etalon.png)

### Three hosts, one floor

The Etalon is the whole preprocessor here: one transit, then the
readout. On spectra like these that is already enough.

Two siblings have now run the same task with the same readout, the
same 60-epoch budget, and the same split. The two-stage
([HypercubeCascade](https://github.com/dliptak001/HypercubeCascade))
puts a frozen HypercubeWTF reservoir behind this very transit and
scores 4.82. The reservoir-only
([HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)) runs
that reservoir alone, with the normalized spectrum as its drive, and
scores 4.76. Three preprocessors that share no mechanism — a
transit, an orbit, and the two in series — carry the same one-layer,
one-channel readout to the same floor, and their overlays are
indistinguishable from the ones shown.

Real spectra, however, are not nearly this clean. Low laser power,
short integration times, and weak scatterers all put noise on the
spectrum, and that is where a baseline extractor has to earn its
keep.

That is where the hosts should separate. The Cascade's MNIST
white-noise study
([`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/mnist/WhiteNoiseFilter.md))
found the two-stage path pulling ahead of the Etalon alone from
moderate noise up, and WTF's own study
([`examples/mnist/WhiteNoiseFilter.md`](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/mnist/WhiteNoiseFilter.md))
found its reservoir a filter that holds accuracy as the noise rises.

That is the next experiment.

The overlay, the training profile, and the three-host comparison are
in
[`examples/RamanBaselineExtraction/`](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/RamanBaselineExtraction/README.md).

Runnable programs live under [`examples/`](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/README.md).

The Raman spectra themselves (about 1 GB) are not in this repository.

## Installation

**Preferred:** install a pre-built wheel from PyPI (no compiler).

```bash
pip install hypercube-etalon
```

```python
import hypercube_etalon as he
print(he.__version__)
```

Package name on PyPI: **`hypercube-etalon`**. Import name:
**`hypercube_etalon`**. Main type: **`he.Etalon`**.

Wheels target Python 3.10–3.14 on common Windows, Linux, and macOS machines.
Runtime dependency: NumPy only.

### From source (full repository)

To compile the extension yourself, clone this **entire** repository (not a
minimal source-only download of the `python/` folder alone — the C++ core and
vendored HypercubeCNN live next to `python/`). You need Python 3.10+, a C++23
compiler, and CMake ≥ 3.20.

```bash
git clone https://github.com/dliptak001/HypercubeEtalon.git
cd HypercubeEtalon/python
pip install .
```

On Windows with CLion’s MinGW, put that compiler’s `bin` folder (and Ninja) on
your `PATH`, then:

```bash
pip install . --no-build-isolation --force-reinstall --no-deps
```

(Exact CLion paths change with the version.) Step-by-step toolchain notes:
[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/Python_SDK.md).

---

## Quick start

You bring each sample as a length-**N** float array (N = 2<sup>dim</sup>). How
you get there — pad an image, reshape a spectrum, invent a layout — is up to
you. This package does not pack 784 pixels or 300 bins for you.

Shapes that matter:

| Array | Shape | Notes |
|-------|-------|-------|
| `fields` | `(count, N)` | one length-N field per row |
| `labels` (classification) | `(count,)` | integer class indices |
| `targets` (regression) | `(count, num_outputs)` | float targets |

```python
import numpy as np
import hypercube_etalon as he

dim = 7
N = 2**dim
rng = np.random.default_rng(0)
fields = rng.standard_normal((200, N), dtype=np.float32)
labels = rng.integers(0, 4, size=200)

et = he.Etalon(
    dim=dim,
    exciter_subcube_dim=5,
    exciter_input_scaling=1.0,
    exciter_weight_scaling=0.15,
    readout_num_outputs=4,
    readout_task="classification",
    readout_epochs=80,
)
et.fit(fields, labels)  # collect_batch + train

print(et.N, et.subcube_dim, et.num_collected)
print(f"train sanity check: {et.accuracy_on_collected():.3f}")
print(et.predict_class(fields[0]), et.predict(fields[0]).shape)

et.save("model.pkl")
loaded = he.Etalon.load("model.pkl")
```

### Step by step (same loop, more control)

```python
et = he.Etalon(
    dim=7,
    exciter_subcube_dim=5,
    exciter_input_scaling=1.0,
    exciter_weight_scaling=0.15,
    readout_num_outputs=4,
    readout_task="classification",
)
et.collect_batch(fields_train, labels_train)
et.train()
logits = et.predict(fields_test[0])       # (num_outputs,) float32
cls = et.predict_class(fields_test[0])    # int
test_acc = et.accuracy(fields_test, labels_test)  # held-out, fresh maps
```

For regression, set `readout_task="regression"` and pass float targets instead
of class labels. Then use `r2_on_collected()` / `r2(fields, targets)` the same
way.

`accuracy_on_collected` and `r2_on_collected` only look at the samples you
already trained on — they are a quick sanity check, not a test score. For real
evaluation, hold fields out and call `accuracy` / `r2` (or `predict` /
`predict_class` yourself).

---

## Features

- **One class** — `hypercube_etalon.Etalon` is the whole product surface
- **Map loop** — `collect` / `collect_batch` → `train` →
  `predict` / `predict_class`
- **`fit`** — clear, collect, and train when your arrays are ready
- **dim 4–12** — field length N = 2<sup>dim</sup>; etalon face
  `exciter_subcube_dim`
- **Two gains** — `exciter_input_scaling` (field → transit) and
  `exciter_weight_scaling` (neighbor-weight amplitude); the demos run ~1.0
  and 0.15–0.5 (the header defaults 0.02 barely drive tanh)
- **Built-in ablation** — `bypass_exciter=True` feeds the raw field to the
  same readout
- **Classification or regression** — `readout_task` fixed at construction
- **Held-out scoring** — `accuracy(fields, labels)` / `r2(fields, targets)`
  map fresh in bulk
- **Bulk calls can parallelize** — `collect_threads` (0 = auto)
- **Inspect a map** — `run(x)` then `last_features()` for gain tuning
- **Save / load** — `save` / `load` (pickle: config + readout weights;
  collected samples are not stored). Optional `save_readout_hcnn_model` /
  `load_readout_hcnn_model` for portable HCNW + arch JSON
- **NumPy float32** — arrays converted for you; prefer contiguous float32

---

## Examples

For a first try, paste the [Quick start](#quick-start) after
`pip install hypercube-etalon`. That is self-contained.

If you want a longer walk-through, the demo scripts on GitHub under
[`python/examples/`](https://github.com/dliptak001/HypercubeEtalon/tree/main/python/examples)
are there to open or download — they are not added to your machine by pip.

| Script | What it is for |
|--------|----------------|
| [synthetic_classification.py](https://github.com/dliptak001/HypercubeEtalon/blob/main/python/examples/synthetic_classification.py) | Multi-class toy fields: `fit`, then train and test accuracy |

```bash
# from a clone of HypercubeEtalon, after: pip install hypercube-etalon
python python/examples/synthetic_classification.py
```

These use easy made-up fields so the API is obvious — not scores to publish.
More notes:
[python/examples/README.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/python/examples/README.md).

---

## Documentation

| Doc | Role |
|-----|------|
| **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/Python_SDK.md)** | Canonical Python API — every method, layout, pickle, limits |
| [python/examples/README.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/python/examples/README.md) | Demo scripts on GitHub |
| [Project README](https://github.com/dliptak001/HypercubeEtalon#readme) | Product story and C++ demos from the repo root |
| [docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/CPP_SDK.md) | Native library guide (same product, C++) |
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/mnist/WhiteNoiseFilter.md) | Early white-noise study (MNIST as a test bed) |

---

## Ecosystem

- **[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF)** — a reservoir orbit as the frozen stage instead of a transit.
- **[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)** — the transit and the orbit in series; Etalon is its first stage.
- **[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN)** — cube-native conv stack; Etalon’s trainable head.
- **[HypercubeESN](https://github.com/dliptak001/HypercubeESN)** — echo-state / reservoir computing on streams.
- **[HypercubeHopfield](https://github.com/dliptak001/HypercubeHopfield)** — Hopfield-style dynamics on the cube.

---

## License

Apache 2.0. See [LICENSE](https://github.com/dliptak001/HypercubeEtalon/blob/main/LICENSE).

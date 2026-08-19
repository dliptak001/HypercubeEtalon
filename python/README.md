# HypercubeEtalon

**HypercubeEtalon** is for high-dimensional data that has no natural clock —
spectra, sensor frames, packed images, stills. Those are the same kinds of
static fields people usually feed a spatial CNN, an MLP, or a similar
feed-forward stack. HypercubeEtalon puts **one frozen hypercube preprocessor**
in front of the CNN: an **etalon transit** — a deterministic wave swept across
every vertex/antipode cavity of a Boolean hypercube. The transit's neighbor
weights are drawn once and never trained. A small
[HypercubeCNN](https://github.com/dliptak001/HypercubeCNN) head trains on the
**transit output**. The CNN never sees the original field — it sees what the
wave leaves behind.

That is the product idea: take a static field, pass it through a frozen
nonlinearity, and train a spatial readout on what remains. The aim is a
preprocessor effective enough that the readout can be a single convolutional
layer with a single channel and no pooling.

This package is the **Python** surface for that product
(`import hypercube_etalon`).
Full API reference: **[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/Python_SDK.md)**.
C++ integration guide: **[docs/CPP_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/CPP_SDK.md)**.
Project home: **[github.com/dliptak001/HypercubeEtalon](https://github.com/dliptak001/HypercubeEtalon)**.

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
  preprocessor reconstructs mathematically.
- **Perfect homogeneity** — every vertex has the same degree and the same local
  world, so local dynamics mean the same thing everywhere — no structural
  favorites baked in by a random graph.
- **Cheap navigation** — each neighbor is a few bit operations on the vertex
  index, not a pointer chase through a stored edge list, so walks stay
  arithmetic and cache-friendly.
- **Topology-native pairing** — the readout consumes the preprocessor output
  with zero geometric distortion, and the learned kernels exploit the same
  locality that generated the dynamics. The data never leaves the hypercube it
  was born on.

Each product in the family is a different architecture on that same foundation.

---

## What is HypercubeEtalon?

An *etalon* here is a start vertex and its face antipode treated as a pair of
reflectors. One transit sweeps a tanh wave out to the antipode and back for
every start vertex on the cube; the value standing at the start after the
wave returns is that vertex's output sample. N starts → N output samples —
the transit output is a field with the same length and vertex indexing as the
input.

In classical reservoir computing (and in the siblings
[HypercubeWTF](https://github.com/dliptak001/HypercubeWTF) and
[HypercubeCascade](https://github.com/dliptak001/HypercubeCascade)):

- Preprocessor weights are **frozen**
- Only a **readout** is trained
- Nonlinear dynamics expand and mix the drive into a rich state

Whether the single-stage transit has **real product value** is still an open
question. Early studies suggest it adds noise filtering the raw field does
not have (see [Early observations](#early-observations-exploratory)), and the
built-in `bypass_exciter` flag runs the no-transit ablation so you can check
on your own task.

---

## Pipeline

```text
x  (your length-N field — already on the cube, no natural time)
    │
    ▼
 frozen etalon transit (one wave over every cavity)
    │
    ▼
 transit output (length N) → HypercubeCNN → logits / values
```

- Cube size from **dim** (N = 2<sup>dim</sup>; dim 4…12, prefer ≥ 5).
- Only the readout trains.
- Everyday loop in this package:
  `collect_batch` → `train` → `predict` / `predict_class`,
  or one-shot `fit` (collect + train).

Unlike HypercubeESN’s Python API, there is no stream of small samples over real
time and no next-step `fit` on a 1D signal. Each sample is one full field; the
CNN only ever sees the transit output for that field.

Full method list and knobs:
**[docs/Python_SDK.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/docs/Python_SDK.md)**.

---

## Early observations (exploratory)

On the MNIST white-noise study (train clean, test with Gaussian field noise),
the transit path holds its accuracy markedly better than the pack-only bypass
as test noise grows. On a Raman baseline-extraction regression the etalon
readout runs at the goal size — one conv layer, one channel, no pooling — and
edges out the two-stage cascade sibling on validation RMSE. The write-ups
have the details and how we ran them:

| Document | Question |
|----------|----------|
| [WhiteNoiseFilter.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/mnist/WhiteNoiseFilter.md) | Noisy test fields: does the transit help vs pack-only → CNN? |
| [RamanBaselineExtraction/README.md](https://github.com/dliptak001/HypercubeEtalon/blob/main/examples/RamanBaselineExtraction/README.md) | Baseline regression at the goal readout size |

The MNIST study uses small cubes because they are handy to pack and run, not
because we are chasing digit accuracy. A more rigorous study is still needed
before treating any of those results as settled. You can reproduce the same
ideas from Python with this package (pack fields yourself, then collect,
train, and predict). The original write-ups and C++ demos that produced the
numbers live under
[`examples/`](https://github.com/dliptak001/HypercubeEtalon/tree/main/examples).

---

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

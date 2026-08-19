# HypercubeEtalon C++ SDK

You place a fixed pattern on the hypercube — an image pack, a spectrum, or any
field you built yourself. HypercubeEtalon **runs one etalon transit over that
field** — a deterministic wave swept through every vertex/antipode cavity of
the cube — then **trains a small CNN on the transit output**. One class does
the whole loop: collect samples, train the head, predict.

You do not need to learn HypercubeWTF, HypercubeCascade, or HypercubeCNN
first. Link **`HypercubeEtalonCore`**, include **`Etalon.h`**, and work with
**`Etalon`**. Demos and packing helpers are optional recipes; they are not the
product.

This guide matches the public headers for **1.0.x**.

**Who it is for:** anyone embedding the Etalon in a host (collect → train →
predict), and anyone learning the stack with the same API the demos use.

**What you get:** a C++23 static library. Headers sit at the repo root. A
vendored HypercubeCNN builds the trainable readout; hosts usually never call
HCNN themselves.

| Section | |
|---------|--|
| [1. Why explore HypercubeEtalon](#1-why-explore-hypercubeetalon) | Family role, what the transit buys |
| [2. The big picture](#2-the-big-picture) | Where Etalon sits among WTF / Cascade / CNN |
| [3. One sample](#3-one-sample-step-by-step) | The transit, mechanics, API words |
| [4. Product surface](#4-what-is-the-product-and-what-is-not) | Headers, rules, the loop |
| [5. Build](#5-build-and-consume) | CMake, binaries |
| [6. First program](#6-first-program) | Minimal collect → train → predict |
| [7. API](#7-the-api-you-actually-use) | Config and methods |
| [8–13](#8-please-do-not) | Boundaries, demos, pitfalls, cheat sheet |

---

## 1. Why explore HypercubeEtalon

HypercubeEtalon processes spatial data through **one frozen preprocessing
stage** in front of a small trainable CNN: an etalon transit. The transit's
neighbor weights are drawn once at construction and never trained. Only the
CNN readout learns.

The point of the experiment is to see whether that frozen stage makes the
readout's job easier — better accuracy at the same readout size, or the same
accuracy at a much smaller readout — than feeding the CNN the raw field
(`bypass_exciter` runs exactly that ablation). On the MNIST white-noise study
the transit path holds up markedly better than the bypass as test noise
grows; see [WhiteNoiseFilter.md](../examples/mnist/WhiteNoiseFilter.md) for
the numbers.

The aim is a preprocessor effective enough that the readout can be a single
convolutional layer with a single channel and no pooling — fast to train,
small in memory, and requiring essentially no CNN architecture engineering.
The Raman baseline example already runs at exactly that readout size.

Whether the single-stage transit has real product value is still an open
question.

---

## 2. The big picture

Most learning systems either see a **stream** (one small input every step) or
a **static pattern** (classify an image once). The Etalon takes a static
pattern and sweeps a deterministic wave through it:

1. You give it one full-length field on the cube (packed however you like).
2. The **Exciter** runs one etalon transit — same field length in, same out.
3. It trains a CNN on that transit output.

The preprocessor never trains. Only the readout does. That is the whole
product idea.

| Library | You typically feed it… | What runs |
|---------|------------------------|-----------|
| **HypercubeEtalon** | a **static** length-N field | **etalon transit → HCNN** |
| [HypercubeWTF](https://github.com/dliptak001/HypercubeWTF) | a **static** length-N field | reservoir orbit → end state → HCNN |
| [HypercubeCascade](https://github.com/dliptak001/HypercubeCascade) | a **static** length-N field | transit → orbit → end state → HCNN |

### Your data does not have to be a power of two

**dim** is the size knob for the whole pipeline: set `cfg.exciter.dim`
(valid **4…12**; prefer ≥ 5 so a pooled readout has room) and you get
**N = 2<sup>dim</sup>** vertices. Leave `cfg.readout.dim` at 0 — construction
fills it from the Exciter; any other value must equal `exciter.dim` or the
constructor throws. The Etalon always expects a field of length N.

If your raw data is 784 pixels or 300 bins, **you** map it onto N floats first
(pad, resize, spatial embed, custom layout — your choice). The Etalon does
not invent that map. Demo helpers under `examples/common/` are one
MNIST-oriented recipe; skip them when you pack your own way.

### What freezes vs what learns

| Piece | Trains? |
|-------|---------|
| Exciter neighbor weights | No — drawn once at construct |
| The input gain (`exciter.input_scaling`) | No — a config scalar you set |
| How you pack domain data into the field | Your problem (outside the Etalon) |
| HCNN readout | **Yes** |

If you change only `exciter.seed`, the frozen transit weights change — and on
the tasks measured so far, the results barely move (seed is not a tuning
parameter). The gains (`input_scaling`, `weight_scaling`) **are** tuning
parameters: they set how hard the tanh sites are driven.

---

## 3. One sample, step by step

Think of one map as: scale the field once, sweep a wave through every cavity
of the cube, keep what stands at each start vertex when its wave comes back.

### The etalon transit

The Exciter is **not** a reservoir: no leak, no delay line, no orbit. It is a
fixed nonlinear map. At construction it draws one weight per (vertex,
neighbor) pair — `N × dim` weights from U(-1, 1) × `weight_scaling` — and
never updates them again.

An *etalon* here is a start vertex `r` and its face antipode treated as a
pair of reflectors. The face is the subcube spanned by the low `subcube_dim`
bits — `M_walk = 2^subcube_dim` vertices, with the high bits pinned by `r`.
One transit scales the input once (by `exciter.input_scaling`), then for
every start `r` reloads that scaled field, walks the face out to the antipode
and back, and writes one output sample: the value standing at `r` after the
second visit. Each site update is tanh of a full-star neighbor sum, applied
sequentially — the disturbance propagates through the face and reflects back.
Off-face neighbors are never updated during a walk, so every update also
mixes in unmodified input.

`N` starts → `N` output samples → the transit output is a field with the same
length and vertex indexing as the input.

### Mechanics in order

1. Copy the caller's field (the Etalon never writes your buffer).
2. One etalon transit over the copy (the copy is scaled in place once, then
   swept start by start).
3. The transit output (length N) **is** the feature row the CNN sees.
4. Hand those features to the readout (collect, train, or predict).

With `bypass_exciter = true`, step 2 is skipped and the features are a plain
copy of the field — the ablation path.

```text
x  (length N, fixed for this sample)
    │
    ▼
 Etalon transit  (or a plain copy, when bypass_exciter)
    │
    ▼
 features (N)
    │
    ▼
 HCNN readout → class logits or regression values
```

### Words you will see in the API

| Word | Plain meaning |
|------|---------------|
| **dim** | Cube dimension you choose (4…12) via `exciter.dim`; `readout.dim` auto-fills |
| **N** | Field length = 2<sup>dim</sup> (input and features — both N) |
| **subcube_dim** | Face size of one etalon cavity; walk covers 2<sup>subcube_dim</sup> vertices |
| **input_scaling** | Gain applied once to the field before the transit |
| **weight_scaling** | Neighbor weights are U(-1, 1) × this |
| **bypass_exciter** | Skip the transit; features = the field (ablation) |

Unlike HypercubeWTF there is no `T` / `B` / `readout_slices` machinery: there
is no orbit, the readout always sees exactly the transit output, and the
feature size is always **N**.

---

## 4. What is the product (and what is not)

| You care about… | Use… |
|-----------------|------|
| Integrating the library | **`Etalon`** + **`EtalonConfig`** |
| Walk size, saving readout weights | `et.exciter()` / `et.readout()` |
| Feature probe | `et.LastFeatures()` |
| Learning by example | `etalon_synth`, `etalon_mnist`, `etalon_raman` |
| MNIST paths / packing demos | `examples/common/` (optional) |
| Raw HypercubeCNN | Almost never — that lives under the readout |

```text
Etalon.h           front door (Etalon + EtalonConfig)
Exciter.h          ExciterConfig (+ Exciter for inspection)
Readout.h          ReadoutConfig, enums, Readout
… .cpp files …
third_party/HypercubeCNN/    vendored; see VENDORED.md
examples/                    demos, not the SDK definition
docs/CPP_SDK.md              this guide
```

Link **`HypercubeEtalonCore`** (it pulls **`HypercubeCNNCore`** for you).

### Rules that matter

These are product contracts, not implementation trivia.

- **Every field is length N.** Wrong size throws.
- **Values are usually kept in [-1, 1].** The library trusts the host; it does not clamp.
- **You pack; the Etalon maps.** No built-in image layout.
- **One dim.** Set `exciter.dim`; leave `readout.dim` at 0 (auto). A nonzero
  mismatch throws.
- **The Exciter freezes at construct.**
- **`Predict` returns raw logits** (or regression values) — no softmax.
- **`AccuracyOnCollected` / `R2OnCollected` are training-set scores.** For
  held-out data use `Accuracy(fields, labels)` / `R2(fields, targets)`, which
  map fresh.
- **There is no built-in train-input noise knob.** If a study needs noisy
  collect or noisy eval, the host adds the noise (see `etalon_mnist`).
- **One `Etalon` per thread of control.** Bulk collect parallelizes *inside*
  one call; do not call public methods concurrently on the same object.
- **`Etalon` is not copyable but is movable.** Moved-from objects may only be
  assigned to or destroyed.

### The loop you will write

```text
fill EtalonConfig
construct Etalon once
collect many samples      (Collect / CollectBatch)
TrainOnCollected
Predict / PredictClass    (always a fresh clean map)
```

Optional extras: `Run` + `LastFeatures`, train-set metrics, held-out
`Accuracy` / `R2`, `ClearCollected`, `collect_threads` for faster bulk
collect, `bypass_exciter` for the ablation.

---

## 5. Build and consume

You need **C++23** and **CMake ≥ 3.21**. Prefer **Release** when you care
about study numbers (Debug and Release float behavior can differ with this
project's fast-math flags).

In CLion: open the project, reload CMake, build. From a shell with the
toolchain available:

```bash
cmake --build cmake-build-release
```

When this repo is the top-level project you also get:

| Binary | Role |
|--------|------|
| `HypercubeEtalon` | Contract smoke (sizes, determinism, bypass, move) |
| `etalon_synth` | Multi-class synthetic fields (no data files) |
| `etalon_mnist` | MNIST recipe + test-noise sweep (IDX files under `C:\HypercubeEtalon\data`) |
| `etalon_raman` | Raman baseline regression (spectra under `C:\HypercubeEtalon\RamanSpectraLCOHard`) |
| `etalon_raman_extract` | Writes selected baseline extracts from a saved readout |

If you pull HypercubeEtalon in as a **subdirectory**, demos are skipped; you
still get the library.

```cmake
add_subdirectory(path/to/HypercubeEtalon)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE HypercubeEtalonCore)
```

```cpp
#include "Etalon.h"
```

(`HypercubeEtalonCore` exports the repo root as a public include directory,
so the one include line is enough.)

---

## 6. First program

A tiny two-class example — collect, train, predict. (Verified against the
library: trains and predicts correctly on this toy task.)

```cpp
#include "Etalon.h"
#include <cstdio>
#include <vector>

int main() {
    EtalonConfig cfg;
    cfg.exciter.dim = 5;                // N = 32; readout.dim auto-fills
    cfg.exciter.subcube_dim = 4;        // default 6 is illegal at dim 5
    cfg.exciter.seed = 1;
    cfg.exciter.input_scaling = 1.0f;   // defaults (0.02) barely tickle tanh
    cfg.exciter.weight_scaling = 0.5f;

    cfg.readout.num_outputs = 2;
    cfg.readout.task = ReadoutTask::Classification;
    cfg.readout.epochs = 80;
    cfg.readout.conv_channels = 4;
    cfg.readout.num_threads = 1;

    Etalon et(cfg);
    const size_t N = et.N();

    auto field = [&](int label) {
        std::vector<float> x(N, 0.f);
        const float s = (label == 0) ? 1.f : -1.f;
        for (size_t i = 0; i < N / 2; ++i)
            x[i] = s * (0.2f + 0.8f * float(i) / float(N));
        return x;
    };

    for (int i = 0; i < 24; ++i) {
        et.Collect(field(0), 0);
        et.Collect(field(1), 1);
    }
    et.TrainOnCollected();

    // AccuracyOnCollected is the *training* set, not test data.
    std::printf("train acc=%.3f  pred0=%d pred1=%d\n",
                et.AccuracyOnCollected(),
                et.PredictClass(field(0)),
                et.PredictClass(field(1)));
    return 0;
}
```

**Habits that save pain later**

1. Build the config, construct **one** `Etalon` (weights freeze here).
2. Collect a dataset, then train. You can call `TrainOnCollected` again
   without clearing — it continues from the current readout weights.
3. Treat `Predict` / `PredictClass` as clean inference — each is a fresh map.
4. Keep packing in your code (or a demo helper). The core only accepts
   length-N fields.

---

## 7. The API you actually use

Authoritative signatures and contracts live in **`Etalon.h`** (and the
headers it pulls). This section is the host-oriented map.

### Config at a glance

Everything interesting is set **before** `Etalon` is constructed.

```cpp
struct EtalonConfig {
    ExciterConfig exciter{};         // exciter.dim is the size knob [4, 12]
    ReadoutConfig readout{};         // leave readout.dim = 0 (auto)

    bool bypass_exciter = false;     // features = a copy of the field (ablation)
    size_t collect_threads = 0;      // bulk maps: 0 = auto, 1 = serial, K = K
};
```

**Exciter** (frozen transit) — the preprocessing knobs:

| Field | Meaning | Valid / notes |
|-------|---------|---------------|
| `dim` | Cube dimension; N = 2<sup>dim</sup> | **[4, 12]**; prefer ≥ 5 for pooled readouts |
| `seed` | Neighbor-weight draws | any `uint64_t`; results are seed-insensitive |
| `input_scaling` | Scales the field once before the transit | demos use 1.0 (the header default 0.02 barely drives tanh) |
| `weight_scaling` | Neighbor weights are U(-1, 1) × this | demos use 0.15…0.5 |
| `subcube_dim` | Etalon face size; walk covers 2<sup>subcube_dim</sup> vertices | **[1, dim]** — the default 6 throws below dim 6 |

**Readout** (trainable head) — knobs most hosts touch:

| Field | Meaning |
|-------|---------|
| `num_outputs` | Classes, or regression width |
| `task` | `ReadoutTask::Classification` or `Regression` |
| `epochs`, `batch_size` | Batch training |
| `lr_max`, `lr_min_frac`, `lr_decay_epochs` | Cosine learning-rate schedule |
| `num_layers`, `conv_channels`, `channel_growth` | Conv stack size (the goal is 1 / 1 / 1) |
| `use_pooling`, `pool_type` | Antipodal pool after each conv |
| `activation` | Per-conv activation (demos often `NONE`) |
| `num_threads` | HCNN workers — use **1** for simple determinism |
| `restore_best_epoch` | Keep best-epoch weights (default true) |
| `seed` | Readout weight init |

Deeper fields (batch-norm, optimizer, holdout fraction, `epoch_tick`) live on
`ReadoutConfig` in `Readout.h`.

### After construct — sizes and inspection

```cpp
explicit Etalon(const EtalonConfig& cfg);

et.Dim();  et.N();
et.NumCollected();
et.CollectedFeatures();    // span, sample-major, NumCollected() * N
et.NumOutputs();
et.CollectThreads();       // configured preference (0 = auto)

et.config();               // resolved knobs (readout.dim filled in)
et.exciter();              // const — transit config, walk size
et.readout();              // mutable — weights, HCNW save/load, IsTrained
```

Construction checks the usual mistakes: dim out of [4, 12], `subcube_dim` out
of [1, dim], a nonzero `readout.dim` that does not match, `num_outputs < 1`,
plus every stage's own checks.

### Run a map (no training)

```cpp
et.Run(x);                         // x.size() == N; x is not modified
auto f = et.LastFeatures();        // length N — what the readout would see
```

`LastFeatures` updates on every completed map on the instance — `Run`,
`Collect`, `Predict`, `PredictClass`, `Accuracy`, `R2`, and bulk collects
(last row). The span is valid until the next map — copy what you keep.

The demos use this probe to print mean-|value| lines ("~1 is a live field,
~0 is crushed") — the fastest way to tune the two exciter gains.

### Collect, train, predict

**Classification**

```cpp
et.Collect(x, class_label);               // one sample
et.CollectBatch(fields_flat, labels);     // bulk, sample-major
```

**Regression** — same idea with target vectors (`num_outputs` floats per
sample):

```cpp
et.Collect(x, targets);
et.CollectBatch(fields_flat, targets_flat);
```

```cpp
et.ClearCollected();             // drop training rows (keeps worker pool)
et.TrainOnCollected();           // needs at least one sample; does not clear

auto out = et.Predict(x);        // num_outputs floats; no softmax
int y = et.PredictClass(x);      // classification only

double acc = et.AccuracyOnCollected();    // train set
double r2  = et.R2OnCollected();          // train set, regression
```

**Held-out scoring** — these map every field fresh (bulk, parallel), then
score:

```cpp
double test_acc = et.Accuracy(test_fields_flat, test_labels);
double test_r2  = et.R2(test_fields_flat, test_targets_flat);
```

Bulk layout notes:

- `fields_flat` is sample-major: sample `i` starts at `i * N`.
- Labels are validated **before** any mapping starts; a throw leaves the
  collected set unchanged.
- Wrong task (class API on a regression net, etc.) throws.

### Faster bulk collect

`EtalonConfig::collect_threads`:

- `0` — auto (leaves one or two cores free so the machine stays responsive)
- `1` — serial
- `K` — up to K workers

Worker 0 reuses the primary Exciter. Extra workers clone the frozen Exciter
once (same seed → identical weights) — unless `bypass_exciter`, where there
is nothing to clone. The internal thread pool **grows** for the life of the
`Etalon` and does not shrink. Single-sample calls are always serial.

---

## 8. Please do not

| Temptation | Better path |
|------------|-------------|
| Drive `Exciter` yourself for product training | Use `Etalon` maps |
| Call vendored `hcnn::HCNN` from the app | Let `Readout` own it; export via `et.readout()` if needed |
| Depend on `examples/common` in production | Copy the idea; own your packing |
| Chain a second frozen stage by hand | That experiment exists: [HypercubeCascade](https://github.com/dliptak001/HypercubeCascade) |
| Size the readout separately | Leave `readout.dim = 0`; construction fills it |

`Exciter` and `Readout` headers are public so config and inspection work. The
happy path is still collect → train → predict on **`Etalon`**.

---

## 9. Demos as recipes

Demos keep product knobs in `MakeBaseConfig()` / `MakeConfig()` and demo-only
constants (`k*`) beside them:

```text
config → Etalon → collect → train → score → predict
```

| Demo | When to open it |
|------|-----------------|
| `examples/synth/etalon_synth.cpp` | Fast multi-class gate without data files |
| `examples/mnist/etalon_mnist.cpp` | Real packing, bypass comparison, test-noise sweep |
| `examples/RamanBaselineExtraction/etalon_raman.cpp` | Regression at the goal readout size, save + reload check |
| `examples/RamanBaselineExtraction/etalon_raman_extract.cpp` | Inference-only from a saved readout |

More context: [`examples/README.md`](../examples/README.md).

---

## 10. Threads, memory, and cost

- Treat one `Etalon` as **exclusive** for public calls.
- Bulk collect is where parallelism belongs; it clones the frozen Exciter as
  needed.
- Per-sample cost is the transit:
  `N × (2 × 2^subcube_dim − 1) × dim` multiply-adds, plus an N-float reload
  per start. There is no orbit — one map here is the cheap end of the family.
- Features are always `N` floats; the CNN scales with its own layers and
  channels.
- If you already run many `Etalon` instances in parallel, set
  `readout.num_threads = 1` so HCNN does not oversubscribe the machine.
- Prefer Release when comparing accuracies across runs.

---

## 11. Common mistakes

| Symptom / assumption | Fix |
|----------------------|-----|
| Throw on collect / run | Field length must equal `et.N()` |
| "Why can't I pass 784 floats?" | Pack to N first |
| Construct throws at small dim | Default `exciter.subcube_dim = 6` needs dim ≥ 6 — set it ≤ dim |
| Construct throws on `readout.dim` | Leave it 0 (auto); nonzero must equal `exciter.dim` |
| Softmax inside `Predict` | You get logits; use `PredictClass` or argmax |
| Expected a train-noise σ knob | Not in this host — add noise in your collect loop |
| Great train accuracy, bad real test | `AccuracyOnCollected` is the training set; use `Accuracy(...)` on held-out fields |
| Features look crushed (~0) | Header defaults drive tanh weakly — the demos run `input_scaling` ≈ 1.0 and `weight_scaling` 0.15…0.5; probe with `Run` + `LastFeatures` |
| Racey results with shared `Etalon` | One instance, one host thread of control |
| Linked HypercubeCNN only | Link `HypercubeEtalonCore`, include `Etalon.h` |

---

## 12. Further reading

| Doc | What it is |
|-----|------------|
| [WhiteNoiseFilter.md](../examples/mnist/WhiteNoiseFilter.md) | White-noise study: transit vs bypass |
| [RamanBaselineExtraction/README.md](../examples/RamanBaselineExtraction/README.md) | Regression task and results |
| [examples/README.md](../examples/README.md) | Demo map and data-file notes |
| [VENDORED.md](../third_party/HypercubeCNN/VENDORED.md) | Which HypercubeCNN pin is in tree |
| [HypercubeWTF](https://github.com/dliptak001/HypercubeWTF) / [HypercubeCascade](https://github.com/dliptak001/HypercubeCascade) | The reservoir sibling and the two-stage experiment |

---

## 13. Cheat sheet

```text
#include "Etalon.h"

EtalonConfig cfg;
cfg.exciter.dim = 7;                   // N = 128; readout.dim auto-fills
cfg.exciter.subcube_dim = 5;           // <= dim
cfg.exciter.input_scaling = 1.0f;
cfg.exciter.weight_scaling = 0.15f;
cfg.collect_threads = 0;               // auto

cfg.readout.num_outputs = K;
cfg.readout.task = ReadoutTask::Classification;
cfg.readout.epochs = 100;
cfg.readout.num_threads = 1;

Etalon et(cfg);

et.CollectBatch(fields_flat, labels);       // count * N floats, sample-major
et.TrainOnCollected();

int y = et.PredictClass(x);
auto logits = et.Predict(x);
double test_acc = et.Accuracy(test_flat, test_labels);

et.Run(x);                                  // probe one map
auto feats = et.LastFeatures();             // N
```

**In one line:** pack a field → frozen etalon transit → features → train the
CNN head.

# HypercubeEtalon

Experiment: map a length-N field on the Boolean hypercube through a bank of
XOR-rotated nonlinear sweeps, and train a HypercubeCNN readout on the result.

Vendored **HypercubeCNN** (Apache-2.0 core) lives under
`third_party/HypercubeCNN/` — see that tree’s `VENDORED.md`. Host CMake builds
`HypercubeCNNCore` and **`HypercubeEtalonCore`** (`Exciter` + `Readout` +
product façade `Etalon`).

## Idea

Vertices are `0 .. N-1` with `N = 2^dim`. Two vertices are neighbors if their
labels differ by one bit. Each vertex owns `dim` fixed random weights, one per
axis, drawn once at construction (`U(-1,1) * weight_scaling`).

The object that owns those weights and runs the passes is an **Exciter**.
Given an input field `x` of length N, `ExciteCube(x)` does the following for
every rotation index `r = 0 .. N-1`:

1. Scale the field once in place: `x *= input_scaling`.
2. For each rotation `r`, copy that scaled field onto the vertices
   (`state = x`), then run a forward then reverse in-place pass. At each
   vertex the update is a weighted sum over the full hypercube neighbor star
   (all `dim` axes), then `tanh`. Visit order is the standard index order
   XOR-translated by `r` (same local rule from each group translate).

   - **Forward:** `v = 0 .. N-1` (last write is the logical antipode `v = N-1`,
     physical site `(N-1) xor r`).
   - **Reverse:** `v = N-2 .. 0` — turnaround after the forward endpoint; do
     not re-update `v = N-1` on entry into the return leg.

3. Write `output[r] = state[r]` after that pair of passes (value at the frame
   origin after the return leg).

The result is a length-N vector: one excitation sample per rotation.

The dynamics are spatial, order-dependent (async updates), and **highly
initial-condition dependent**: the forward/reverse sweep is a strong mixing of
whatever sits on the vertices at the start of the pass. That is why each
rotation reloads `state` from `x` before running — every `r` must see the same
IC, not the wreckage of the previous rotation. There is no delay line, no
spectral-radius retune, no learned input matrix, and no leak; this is not a
time-stepping reservoir in the usual ESN sense.

## Product API (`Etalon`)

Public entry point is **`Etalon`**: frozen Exciter + trainable Readout.

```text
EtalonConfig cfg;
cfg.exciter.dim = 5;                 // N = 32; product dim is [4, 10], prefer >= 5
cfg.exciter.input_scaling = 1.0f;
cfg.exciter.weight_scaling = 0.5f;
cfg.readout.dim = 0;                 // auto = exciter.dim
cfg.readout.num_outputs = 2;
cfg.readout.task = ReadoutTask::Classification;

Etalon et(cfg);
et.Collect(field, class_label);      // map field → features, append
et.TrainOnCollected();
int cls = et.PredictClass(field);    // fresh map + argmax (no collect noise)
```

Pipeline:

```text
x[N]  →  [optional collect-only noise]  →  Exciter  →  y[N]  →  Readout
                                         (or bypass: y = x)
```

| Method | Role |
|--------|------|
| `Run` | Field → features; updates `LastFeatures`; no train noise |
| `Collect` / `CollectBatch` | Map + append to training set (optional noise) |
| `TrainOnCollected` | Batch-train the readout on collected features |
| `Predict` / `PredictClass` | Fresh map + forward (no train noise) |
| `AccuracyOnCollected` / `R2OnCollected` | Metrics on the **training** set |
| `Accuracy` / `R2` | Fresh map + metric on a caller-owned set (test) |
| `ClearCollected` | Drop the training buffer |

Callers’ field buffers are **never** mutated (`ExciteCube` scales a private copy).

Optional product knobs on `EtalonConfig`:

- `bypass_exciter` — features = field (ablation baseline)
- `train_input_noise_sigma` / `noise_seed` — collect-only Gaussian noise

### Not carried over from HypercubeWTF

No reservoir orbit (`T`), delay-line packing (`B`/`M`), episode IC (`s0`), or
parallel collect pool. One map is one Exciter bank pass; features are always
length N.

### Dimension ranges

| Surface | dim | Why |
|---------|-----|-----|
| Exciter | 4 .. 10 | 4 is the smallest useful star; 10 is the cost cap (N = 1024) |
| Readout | 3 .. 30 | HypercubeCNN hard limit. With pooling, `num_layers <= dim-2` |
| Etalon | 4 .. 10 | Same as Exciter; `readout.dim` auto-matches |

Prefer `dim >= 5` for a roomier default pooled stack. That is guidance, not a
hard floor: dim 4 constructs and trains. Illegal stacks throw
`std::invalid_argument` in both Debug and Release.

### Lower-level pieces

`Exciter` and `Readout` remain usable directly when you do not want the façade.

**Exciter contracts:** length-N non-null field; scales **in place**; return
pointer is owned by the Exciter and invalid after the next `ExciteCube`.

**Readout contracts:** input length `2^dim` with dim in [3, 30]; `Train`
continues from current weights; prefer `SaveHcnnModel` over the unversioned
`Weights` blob for portable checkpoints.

**Readout is frozen.** It is a HypercubeCNN façade so the product can train a
head. Do not add more training-loop policy (new LR schedules, extra
checkpoint schemes, more holdout rules). New work goes on the Exciter map
or the `Etalon` façade. A different train loop should drive HypercubeCNN
directly.

## Examples

`etalon_synth` (no data files) and `etalon_mnist` (IDX from a fixed deploy
dir) live under [`examples/`](examples/README.md). Each scores Exciter vs
bypass on a held-out set. Build Release; run the binary you care about.

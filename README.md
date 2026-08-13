# HypercubeEtalon

You put a number on every corner of a cube. The **Exciter** walks that field
and writes back a new number for each starting corner. A small
**HypercubeCNN** head then learns to classify (or regress) those numbers.

That is the whole product. One map, one trainable head. No reservoir orbit,
no delay line, no leak.

The cube library lives under `third_party/HypercubeCNN/` (Apache-2.0,
release 1.0.3). This repo builds it as `HypercubeCNNCore` and the host as
`HypercubeEtalonCore`: Exciter, Readout, and the `Etalon` wrapper you
actually call.

## The cube

There are N corners, `N = 2ᵈⁱᵐ`. Label them `0 … N−1`. Two corners are
neighbors if their labels differ by a single bit. Flip bit `j` and you
step along one edge.

Each corner keeps `dim` fixed random weights, one per edge, drawn once at
construction from U(−1, 1) and scaled by `weight_scaling`. Those weights
never learn. Only the head does.

`dim` is small. Four is the tiniest cube with a useful star of neighbors.
Twelve is the cost cap (`N = 4096`). Five (`N = 32`) is the usual starting
point: cheap, and roomy enough for the default pooled head. The default
walk is a **half**-cube (`halvings = 1`).

## A bounce

Pick a starting corner `r`. Copy the input field onto the cube. Then walk.

The walk is a **bounce**. It leaves `r`, hits the far corner — the
antipode, every bit flipped — and comes back to `r`. At each corner it
looks at the neighbors, takes a weighted sum, and writes `tanh` of that
sum in place. Order matters: a later corner sees values the earlier ones
just wrote.

By default the walk uses a **half**-cube: `v = 0 … M−1` with
`M = N / 2` and physical corner `v xor r` (so `v = 0` is `r`, and
`v = M−1` is the far corner of that face). Turn around there. Come back
`v = M−2 … 0`. Do not write that far corner a second time on the way in.
Set `halvings = 0` if you want the whole cube (`M = N`).

When you arrive home, keep `state[r]`. That is the sample for this start.

## A bank of them

Do that once for every `r`. Reload the same input each time — otherwise
start 7 would inherit the wreckage of start 6. Scale the input once, up
front, by `input_scaling`.

You get back N numbers: one bounce per corner. That vector is what the
head sees.

This is not a time-stepping reservoir. There is no leak, no spectral
radius, no learned input matrix. It is a spatial mixer, and it is
sensitive to whatever you put on the corners. That is the point of
reloading.

## Smaller walks

A bounce does not need the far corner of the *whole* cube. Half a cube is
still a cube. So is a quarter, or an eighth.

Set `exciter.halvings` to `0, 1, 2, …` (whole, half, quarter, …) and
each bounce walks only `M = N >> halvings` corners: the face through `r`
whose low `dim − halvings` bits are free. It still starts at `r`,
reflects at the far corner **of that face**, and comes home. You still
write one sample per `r`, so the feature vector is still length N. Each
extra halvings cuts the walk in half. `halvings` must be less than
`dim` (at least two corners on the walk). Default is `1` — half the cube.

Starts that share the same high bits walk the same face from different
doors. Cheaper walks, not fewer samples.

The gather is a **full star**. Every edge counts, including edges that
step off the face onto corners this walk never updates. Those sites stay
at the original input — a frozen wall. A face that misses the packed
picture still sees it through that wall.

Parallelizing the N independent starts is a later job.

## Using it

`Etalon` is the front door: frozen Exciter, trainable head.

```text
EtalonConfig cfg;
cfg.exciter.dim = 5;                 // N = 32
cfg.exciter.halvings = 1;            // 0 = whole cube; 1 = half (default); 2 = quarter
cfg.exciter.input_scaling = 1.0f;
cfg.exciter.weight_scaling = 0.5f;
cfg.readout.dim = 0;                 // auto: same as the Exciter
cfg.readout.num_outputs = 2;
cfg.readout.task = ReadoutTask::Classification;

Etalon et(cfg);
et.Collect(field, class_label);
et.TrainOnCollected();
int cls = et.PredictClass(field);
```

```text
x[N]  →  [optional collect-only noise]  →  Exciter  →  y[N]  →  Readout
                                         (or skip: y = x)
```

| Call | What it does |
|------|----------------|
| `Run` | Map one field. No noise. Updates `LastFeatures`. |
| `Collect` / `CollectBatch` | Map (optional noise) and keep the sample. `CollectBatch` fans independent maps across `collect_threads` workers (0 = auto). |
| `TrainOnCollected` | Fit the head on what you collected. |
| `Predict` / `PredictClass` | Fresh map, then the head. No collect noise. |
| `AccuracyOnCollected` / `R2OnCollected` | Score the **training** buffer. |
| `Accuracy` / `R2` | Fresh map and score a set you pass in (test). |
| `ClearCollected` | Drop the training buffer. |

Your field is never overwritten. `bypass_exciter` skips the walk and
hands the raw field to the head (the fair baseline).
`train_input_noise_sigma` adds Gaussian noise on collect only, never on
`Run` / `Predict`.

`AccuracyOnCollected` is not a test number. Use `Accuracy` on held-out
fields for that.

The head itself is `et.readout()`. Save and load, weight blobs, architecture
summary, and “has this been trained?” all live there — not on `Etalon`.

```text
et.readout().SaveHcnnModel("out/readout");
et.readout().LoadHcnnModel("out/readout");
et.readout().IsTrained();
```

## Sizes

| Piece | dim | Why |
|-------|-----|-----|
| Exciter | 4 … 12 | Smallest useful star … cost cap (`N = 4096`); default `halvings = 1` |
| Readout | 3 … 30 | What HypercubeCNN allows. With pooling, `num_layers ≤ dim−2` |
| Etalon | 4 … 12 | Matches the Exciter; `readout.dim` fills in if you leave it 0 |

Prefer `dim ≥ 5` so a pooled head has room. Dim 4 still builds. Bad
stacks throw `std::invalid_argument` in Debug and Release.

You can use `Exciter` and `Readout` without `Etalon`. `ExciteCube` scales
its buffer in place and returns a pointer that dies on the next call.
`Train` continues from the current weights; for files, prefer
`SaveHcnnModel` over the unversioned `Weights` blob.

The Readout is a thin HypercubeCNN wrapper so this repo can train a head.
It is frozen: no new learning-rate schedules, checkpoint schemes, or
holdout rules. New work belongs on the Exciter or on `Etalon`. A
different train loop should call HypercubeCNN itself.

This is not HypercubeWTF. No orbit length `T`, no delay packing, no
episode initial condition. One map is one bank of bounces. Bulk collect
fans independent maps across workers (`collect_threads`). Features are
always length N.

## Examples

[`examples/`](examples/README.md) has two programs that run the Exciter
on a held-out set. Set `kRunBypass` in a demo if you want the skip-the-walk
check.

- `etalon_synth` — six made-up classes, no data files
- `etalon_mnist` — packed digits, from a fixed folder (see that README)

Build **Release**, then run the binary you care about.

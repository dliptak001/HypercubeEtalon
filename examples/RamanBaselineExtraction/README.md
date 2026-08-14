# RamanBaselineExtraction

Regression example: take a Raman spectrum and estimate its **baseline**.

The cube is **dim 11** (`N = 2048`). Input and readout target are the same
length. The primary class is `BaselineExtractor`.

The example reads the dataset **in place** from
`C:\MLPlayground\Datasets\data`. It does not copy files into this repo or
into `C:\HypercubeEtalon\data`.

`kTrainSamples` / `kTestSamples` are **overall** prefix counts, numeric
index order (`0`, `1`, `2`, …), not lexical filename order. `0` means the
whole split (10000 train / 1000 validation). A small prefix (`100` / `50`)
is the short demo.

---

## Task

| Role | What |
|------|------|
| Input | Raman spectrum, 2048 amplitudes |
| Target / output | Baseline estimate, 2048 amplitudes |
| Unused for now | Peak files (`X.peaks.txt`) |

The x-axis used here is the sample index **0 … 2047**. Each bin is one cube
corner. Files on disk also ship a shared `xaxis.txt` of real wavenumbers; this
example does not use those values.

---

## Dataset on disk

Fixed root (read in place, not copied):

```text
C:\MLPlayground\Datasets\data\
  Training\
  Validation\
```

| Split | Patterns | Index range |
|-------|----------|-------------|
| Training | 10000 | `0` … `9999` |
| Validation | 1000 | `0` … `999` |

Each pattern `X` is three files. Both splits also have one shared axis file.

```text
X.data.txt      input spectrum
X.label.txt     ground-truth baseline (train / score target)
X.peaks.txt     ignore for now
xaxis.txt       2048 wavenumbers; unused (axis is 0 … 2047)
```

Examples: `0.data.txt`, `0.label.txt`, `42.peaks.txt`.

Indices are contiguous. Training and validation reuse the same numeric names
in their own folders; they are different spectra.

---

## File format

- One line, no header.
- 2048 comma-separated ASCII floating-point amplitudes.
- Same count in `.data`, `.label`, `.peaks`, and `xaxis.txt`.

Checked: `Training/0`, `Training/9999`, `Validation/0`, `Validation/999`,
and both `xaxis.txt` files are all length 2048. Training and validation
`xaxis.txt` match (first ≈ 120.72, last ≈ 796.94).

---

## Scale

Same per-spectrum min/max as the C# trainer (from the **input**, never
the label), then shifted from their `[0, 1]` to **[-1, 1]**:

```text
range = max - min
u     = (x - min) / range  // C# [0, 1]
norm  = 2 * u - 1          // here [-1, 1]
x     = (norm + 1) * 0.5 * range + min

If the spectrum is flat, range is 0, norm is 0, and denorm is min.
```

The label uses the **same** min/range as its matching `.data` spectrum, not
its own min/max. Predict only has the input, so the scale has to come from
there.

Collect and train see only these normalized values. `Predict` denormalizes
before it returns.

---

## Error

Score is **RMS** of the denormalized prediction vs raw `X.label.txt`.
No curve fit.

Per spectrum, over the 2048 bins:

```text
err[i]  = label[i] - predicted[i]
RMSE    = sqrt( mean( err[i]^2 ) )
```

On a split (train prefix or validation prefix), take the mean of those
per-spectrum MSEs, then sqrt — same as RMSE over every bin in the split.

That is the number this example reports — the same denormalized RMS as
the C# trainer (`ComputeRMS` / epoch callback). Each readout epoch prints
train `train_rmse`. After fit, the same score is printed on the train
prefix and the validation prefix. Peaks and percent-of-peak error are
out of scope.

---

## Exciter vs bypass (20 epochs)

One Release `etalon_raman` pair after the `[-1, 1]` map. Same 1000 / 100
prefix, dim 11, `halvings = 7` (`M = 16`), 20 readout epochs, activation
NONE. The only change is `kRunBypass`:

```text
bypass=false  — normalized field through the Exciter, then the head
bypass=true   — normalized field straight to the head
```

| Path | train RMSE | val RMSE |
|------|-----------:|---------:|
| Exciter | 42.447 | 42.898 |
| Bypass | 106.832 | 101.552 |

The walk cuts error by about **2.4×**. Train matches val on both arms,
so this is not a memorized 1000. The C# 1-D CNN typically needs about
100 epochs to get under RMSE 3 on this score; these numbers are the
20-epoch cut, not that floor.

# RamanBaselineExtraction

Regression example: take a Raman spectrum and estimate its **baseline**.

The cube is **dim 11** (`N = 2048`). Input and readout target are the same
length. The primary class is `BaselineExtractor`.

This example is a shell so far. Data format is below; the train / score loop
is not written yet.

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

## Error

Score is **RMS** of predicted baseline vs `X.label.txt`. No curve fit.

Per spectrum, over the 2048 bins:

```text
err[i]  = label[i] - predicted[i]
RMSE    = sqrt( mean( err[i]^2 ) )
```

On a split (train prefix or validation prefix), take the mean of those
per-spectrum MSEs, then sqrt — same as RMSE over every bin in the split.

That is the number this example reports. Peaks and percent-of-peak error
are out of scope.

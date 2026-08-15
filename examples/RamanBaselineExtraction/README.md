# RamanBaselineExtraction

A Raman spectrum is a line of 2048 amplitudes: sharp molecular peaks
sitting on a slow, unwanted background. This example asks the Etalon
to estimate that background.

The cube is the same length as the spectrum (`N = 2048`, dim 11), so
each bin is already one vertex. The wrapper class is
`BaselineExtractor`.

The program reads the dataset **in place** from
`C:\HypercubeEtalon\RamanSpectra`. The spectra are about 1 GB and are
not in this repository. They are not the MNIST files under
`C:\HypercubeEtalon\data`.

`kTrainSamples` / `kTestSamples` are **overall** prefix counts, numeric
index order (`0`, `1`, `2`, …), not lexical filename order. `0` means the
whole split (10000 train / 1000 validation). A small prefix (`100` / `50`)
is the short demo.

Shipped `MakeConfig()` uses `subcube_dim = 5` (`M = 32`). The score
tables below were logged at `subcube_dim = 4` (`M = 16`).

---

## Task

| Role | What |
|------|------|
| Input | Raman spectrum, 2048 amplitudes |
| Target / output | Baseline estimate, 2048 amplitudes |
| Unused for now | Peak files (`X.peaks.txt`) |

The x-axis used here is the sample index **0 … 2047**. Each bin is one
vertex. Files on disk also ship a shared `xaxis.txt` of real
wavenumbers; this example does not use those values.

---

## Dataset on disk

Fixed root (read in place, not copied):

```text
C:\HypercubeEtalon\RamanSpectra\
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

---

## Scale

Per-spectrum min/max from the **input**, never the label, mapped to
**[-1, 1]**:

```text
range = max - min
u     = (x - min) / range
norm  = 2 * u - 1
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

That is the number this example reports. Each readout epoch prints
train `train_rmse`. After fit, the same score is printed on the train
prefix and the validation prefix. Peaks and percent-of-peak error are
out of scope.

---

## Exciter vs bypass (20 epochs)

Two Release runs, same 1000 training spectra and 100 validation
spectra. Both used the `[-1, 1]` map, twenty readout epochs, and no
activation. The walk was the shorter one logged above
(`subcube_dim = 4`). The only difference was whether the field went
through the Exciter or straight to the head.

| Path | train RMSE | val RMSE |
|------|-----------:|---------:|
| Exciter | 42.447 | 42.898 |
| Bypass | 106.832 | 101.552 |

The walk cuts error by about **2.4×**. Train matches val on both arms,
so this is not a memorized 1000. These numbers are the 20-epoch cut,
not a claimed floor.

---

## Seed

HypercubeEtalon is highly insensitive to the Exciter seed. That is a
feature: seed is not a tuning parameter.

Three independent `exciter.seed` values, same everything else, full
split (10000 train / 1000 validation), denormalized val RMSE:

| val RMSE |
|---------:|
| 6.155 |
| 6.147 |
| 6.184 |

Spread is 0.037 on a mean of 6.162 (about 0.6%). HypercubeWTF is quite
sensitive to the reservoir seed; here there is one less knob to worry
about.

---

## Best run so far (full split)

Release `etalon_raman`, 10000 train / 1000 validation. Header as printed:

```text
exciter: dim=11 N=2048 subcube_dim=4 M=16 seed=3458567978345987
         in_scale=1 wt_scale=0.15
readout: dim=11 layers=1 conv_channels=1 use_pooling=false pool_type=max
         activation=none epochs=20 batch=48 lr_max=0.003
```

| | |
|--|--:|
| train RMSE | 4.728 |
| val RMSE | 4.713 |
| best epoch | 20 / 20 |
| wall time | 738.80 s |

Train matches val. Early Adam wobble through epoch 8 (`7.82` → `6.76`),
then a clean slide to `4.73`. Last epoch won, so the cosine still had
the best weights at the floor — epochs 17–20 only crawled
(`4.85` → `4.80` → `4.74` → `4.73`).

This is a different operating point from the seed table above (~6.16).
That table is seed-to-seed scatter on one setup; this row is the best
score so far.

---

## Architecture locked

The readout is as tuned as it gets. Swept and discarded as no better
than the current head:

- `num_layers` > 1
- pooling on vs off
- `conv_channels` > 1
- `lr_max` and `lr_min_frac`

Locked readout: 1 layer, 1 channel, pooling off, activation NONE,
batch 48, `lr_max = 0.003`, `lr_min_frac = 0.05`. More epochs will
run later; that is a longer fit, not a new architecture.

That head is already pulling what it can from the Exciter field. The
Exciter knobs (`input_scaling`, `weight_scaling`, `subcube_dim`) are
optimized too — further scalar tweaks will not move the score
meaningfully. Architecture and configs are locked.

The remaining lever is the map itself: multiple Exciter rounds, so
the field gets a better high-dimensional separation of features
before the same readout sees it. Ideas on that next.

---

## Extract and plot

`etalon_raman_extract` loads a saved readout and writes denormalized
baselines for a few dataset indices. `plot_extracted.py` overlays
spectrum, label, and prediction.

Extract knobs live at the top of `etalon_raman_extract.cpp`
(`kDataRoot`, `kSplit`, `kModelStem`, `kRunBypass`, `kIndices`,
`kOutDir`). `kModelStem` is the path without `.hcnw` / `.arch.json`
(example: `C:/HypercubeEtalon/RamanModels/readout_exciter`).
`plot_extracted.py` reads split and indices from `manifest.txt`.
Do not keep a second index list in the plot script.

`MakeConfig()` and `kRunBypass` must match the training run. The
`.arch.json` sidecar checks the readout stack only. The Exciter is
rebuilt from `MakeConfig()` (seed, `subcube_dim`, scales) and is not
in the weight file.

No command-line flags. Knobs are constexpr at the top of
`etalon_raman_extract.cpp`: bypass, data root, split name, model
stem, indices, output directory.

`kIndices` are **dataset file numbers** in that split
(`Training/10.data.txt`, …), not bin positions and not a training
prefix. Change that list to pick different spectra. The extract
program writes them into `manifest.txt`; the plot script follows
the manifest.

When you want the figure:

1. Run `etalon_raman_extract.exe` (same `MakeConfig()` / `kRunBypass`
   as the weights you trained).
2. Run `plot_extracted.py` → `extracted_baselines.png`.

The extract program writes
`C:\HypercubeEtalon\RamanModels\extracted\<idx>.pred.txt`. The plot
script reads those files and writes `extracted_baselines.png` next to
itself.

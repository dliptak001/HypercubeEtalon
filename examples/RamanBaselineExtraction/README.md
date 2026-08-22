# Raman Baseline Extraction

A Raman spectrum is an array of intensity values: sharp molecular
peaks sitting on a slow fluorescence background. The task is to
characterize that background so that it can, in follow-on steps, be
subtracted from the original spectrum, leaving only Raman peaks and
random noise remaining. That extraction is the part conventional
methods fail, often miserably, at. Polynomials, asymmetric least
squares, and ordinary convolutional nets follow the empty stretches
and then ride up into the vibrational excitation bands or cut a hole
under them.

HypercubeEtalon recovers the fluorescence through the peak clusters.

The overlays below come from a sixty-epoch fit on the LCOHard set of
synthetic LiCoO₂ (lithium cobalt oxide) spectra. Ten thousand spectra
for training, two thousand held out. The HypercubeCNN readout that
drew the blue curve is one layer, one channel, no pooling.

Grey is the raw spectrum, red is the true baseline, blue is the
extract. They agree to a few counts (training RMSE 4.71,
validation 4.77) while the fluorescence itself swings by hundreds.
The extract stays on the background instead of climbing into the
bands. All four panels are held-out spectra. The network did not
memorize the training set.

![Held-out validation extract, spectra 581 through 584](extracted_baselines_etalon.png)

Validation/581–584. Blue and red sit on the same slow curve — through
the peak clusters, and down onto the noisy low-count baseline of 584.
The peaks stay in the spectrum.

---

## Results

Full LCOHard split — 10000 training spectra, 2000 held-out validation
spectra. Denormalized RMSE in raw counts (see [Error](#error)).

| Split | RMSE |
|-------|-----:|
| Training | 4.705 |
| Validation | 4.767 |

The validation score sits 0.06 counts above training. At this readout
capacity (one conv layer, one channel, no pooling) the fit does not
overfit the 10000-spectrum split.

Run details, from the run that produced those numbers:

- 60 epochs, batch 48, `lr_max` 0.003, cosine decay,
  `restore_best_epoch` on. Best epoch was 59; the restored weights
  are what both scores measure.
- Collect + train time 2252.0 s on a 32-hardware-thread box
  (`collect_threads = 1`; the HCNN training pool used all 32).

---

## Etalon vs the siblings

Three hosts have now run this task with the same readout shape (one
conv layer, one channel, no pooling), the same 60-epoch budget, and
the same split:

| Host | Preprocessor | Training | Validation |
|------|--------------|---------:|-----------:|
| **HypercubeEtalon** | etalon transit | **4.705** | **4.767** |
| HypercubeCascade | etalon transit → reservoir orbit | 4.779 | 4.819 |
| HypercubeWTF | reservoir orbit | 4.714 | 4.756 |

The Cascade is this Etalon with a frozen HypercubeWTF reservoir
added behind the transit; WTF is that reservoir alone, driven by the
normalized spectrum. The Cascade's Exciter is this one to the seed
(3458567978345987), subcube_dim 5, and input / weight scale
1.0 / 0.15. The Cascade and WTF share their reservoir to the seed
(13871537636959942979), spectral radius 0.95, history depth 8, and
T = 60.

On validation the Etalon lands 0.011 counts above WTF and 0.052
below the Cascade. The three scores span 0.06 counts. At this noise
level that is the spread between hosts, not a ranking: the Etalon
and WTF are a tie, and the Cascade sits a hundredth or so behind
both. The three overlays, on the same four spectra, are
indistinguishable; the Cascade's write-up puts all three side by
side
([HypercubeCascade/examples/RamanBaselineExtraction](https://github.com/dliptak001/HypercubeCascade/blob/main/examples/RamanBaselineExtraction/README.md)),
and WTF's has its own
([HypercubeWTF/examples/RamanBaselineExtraction](https://github.com/dliptak001/HypercubeWTF/blob/main/examples/RamanBaselineExtraction/README.md)).

### Training profile

The three fits earned their scores differently. The Etalon spent
its first third oscillating — 7.00, 7.17, 7.58, 7.31, 6.84, 7.26,
6.89 — and did not descend cleanly until around epoch 14. Both
hosts with a reservoir opened with a fast, clean drop instead: the
Cascade 9.31, 6.90, then 5.87 by epoch 4; WTF 9.40, 6.87, 6.04,
5.81. The Etalon overtook the Cascade on training RMSE at epoch 45
and finished 0.07 counts ahead of it. WTF, after a long shelf
between 5.23 and 5.89 that lasted until epoch 31, descended
steadily to 4.714 at epoch 59 and finished 0.009 counts behind the
Etalon. All three reached their floor in the last few epochs, with
best epoch 59 of 60 and a small tick upward at 60, so none was
still improving meaningfully when it stopped.

---

## Dataset on disk

Fixed root (read in place, not copied):

```text
C:\HypercubeEtalon\RamanSpectraLCOHard\
  Training\
  Validation\
```

| Split | Patterns | Index range |
|-------|----------|-------------|
| Training | 10000 | `0` … `9999` |
| Validation | 2000 | `0` … `1999` |

Each pattern `X` is three files. Both splits also have one shared axis file.

```text
X.data.txt      input spectrum
X.label.txt     ground-truth baseline (train / score target)
X.peaks.txt     ignore for now
xaxis.txt       2048 wavenumbers; ignored by the programs, used by plot_extracted.py
```

Indices are contiguous. Training and validation reuse the same numeric names
in their own folders; they are different spectra.

---

## Host knobs

`MakeConfig()` in `BaselineExtractor.h`. The values behind the
results above:

| Knob | Value |
|------|-------|
| Cube | dim 11, N = 2048 |
| Exciter | subcube_dim 5 (M = 32), input / weight scale 1.0 / 0.15, seed 3458567978345987 |
| Readout | 1 layer, 1 channel, no pooling, no activation, epochs 60, batch 48, lr_max 0.003 |

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

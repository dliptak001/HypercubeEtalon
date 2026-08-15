# HypercubeEtalon examples

Three runnable programs. Each builds a length-N field, sends it through
the Etalon (Exciter then Readout), and scores a held-out set.

Etalon itself is collect, train, predict. These programs own everything
outside that: synthetic patterns, MNIST IDX, and Raman spectra on disk.

| Program | What it is for | Data files? |
|---------|-----------|-------------|
| `etalon_synth` | Fast gate | No |
| `etalon_mnist` | Packed handwritten digits | Yes — see below |
| `etalon_raman` | Raman baseline regression | Yes — not shipped |

Knobs live at the top of each program: the `EtalonConfig` factory
(`MakeBaseConfig()`, or `MakeConfig()` for Raman) and the demo-only
constexprs beside it.

---

## `etalon_synth`

The program invents six classes on a small cube (`N = 128`): two
tones in the low addresses, a handful of signed spikes in the high
addresses, and a little noise. The test set is new draws of those
same classes, not the training fields again.

A run fails if Exciter test accuracy is below 0.70.

---

## `etalon_mnist`

Ten-class handwritten digits. A 28x28 image is 784 pixels; the cube
here is `N = 1024` (dim 10). HypercubeCNN's spatial embed lays the
full digit into the low addresses and a centered crop into the
leftover budget. That packed field is what the Etalon sees.

This example is not chasing a record MNIST score.

### Data setup

`etalon_mnist` does not read MNIST from this git clone. It looks only
at:

```text
C:\HypercubeEtalon\data\
```

Put the four uncompressed IDX files there (see
[Appendix: MNIST files](#appendix-mnist-files)). The dataset is not in
this repository.

---

## `etalon_raman`

A Raman spectrum is a line of 2048 amplitudes: sharp molecular peaks
sitting on a slow, unwanted background. This example asks the Etalon
to estimate that background. The cube is the same length as the
spectrum (`N = 2048`, dim 11), so each bin is already one address —
there is nothing to pack.

This example is still early. The spectra are about 1 GB and are not
yet in the repository.

Task write-up: [`RamanBaselineExtraction/README.md`](RamanBaselineExtraction/README.md).

---

## Folder layout

```text
examples/
  README.md
  common/                 shared helpers (not the core library)
  synth/etalon_synth.cpp
  mnist/                  MNIST demo + IDX loader
  RamanBaselineExtraction/
```

---

## Appendix: MNIST files

**Location:** `C:\HypercubeEtalon\data\`

**Required files** (uncompressed IDX, exact names):

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are the usual public MNIST binaries (LeCun et al.). We do not
ship them in git.

**Download example** (run from `C:\HypercubeEtalon\data`, or save into it):

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

On Windows, any tool that downloads those four `.gz` files and
decompresses them into `C:\HypercubeEtalon\data` is fine.

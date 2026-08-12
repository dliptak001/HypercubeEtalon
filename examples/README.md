# HypercubeEtalon examples

Two runnable demos of the product loop: pack or invent a length-N field,
map it through **Etalon** (frozen Exciter bank + trainable HypercubeCNN
readout), and score a **held-out** set.

The library is collect → train → predict. These programs own everything
outside that: synthetic patterns, MNIST IDX, and how a 28x28 image sits on
the cube.

The product path is the **Exciter**. Bypass (skip the walk) is an optional
consistency check: set `kRunBypass` in the demo `.cpp` when you want it.

Build in **Release**, then run the binary you care about. Debug is slower
and, with `-ffast-math` off, not the number you want to quote.

| Program | What it is for | Data files? |
|---------|----------------|-------------|
| `etalon_synth` | Fast gate, no download | No |
| `etalon_mnist` | Packed handwritten digits | Yes — see below |

Knobs live at the top of each `.cpp`: product options in `MakeBaseConfig()`,
demo-only constants just under that.

---

## `etalon_synth`

Six structured classes on a dim-7 cube (N = 128). Train and test draws use
different `(label, rep)` pairs. Soft floor: Exciter **test** accuracy ≥ 0.70.

Source: [`synth/etalon_synth.cpp`](synth/etalon_synth.cpp).

---

## `etalon_mnist`

Same loop, inputs are MNIST digits packed with HCNN SpatialEmbed
(`PadLowCenter` by default: full 28x28 plus a centered crop in the tail,
which fills N = 1024 at dim 10).

### Defaults are a subset

The shipped constants take **100 train / 50 test per digit** (1000 / 500),
class-balanced, first-in-file-order. That is a demo you can finish, not a
leaderboard run. The soft pass/fail floor is on **Exciter** test accuracy.

A full 60k collect at dim 10 is one Exciter bank pass per image.
Even with the default half-cube walk that is a lot of tanh updates.
Raise `kTrainPerClass` only when you mean to wait.

Do **not** treat the subset score as the best this family can do.
HypercubeCNN alone has already shown about 99.5% on MNIST; this example is
not chasing that.

Source: [`mnist/etalon_mnist.cpp`](mnist/etalon_mnist.cpp).

### Data setup

`etalon_mnist` does **not** read MNIST from this git clone. It looks only at
fixed deploy folders, in order:

```text
C:\HypercubeEtalon\data\
C:\HypercubeWTF\data\
```

Put the four uncompressed IDX files in either place (see
[Appendix: MNIST files](#appendix-mnist-files)). The dataset is **not** in
this repository. If you already have the WTF copy, you do not need a second.

### What was not ported

`wtf_mnist` carried geometric aug, test-field AWGN, and two study write-ups
about the reservoir orbit as a noise filter. Those questions are not restated
here. Re-open them after Exciter scales and cost are settled.

---

## Folder layout

```text
examples/
  README.md
  common/                 shared helpers (not the core library)
  synth/etalon_synth.cpp
  mnist/                  MNIST demo + IDX loader
```

To add another demo: `examples/<name>/`, a target in the root
`CMakeLists.txt`, reuse `common/` if it helps.

---

## Appendix: MNIST files

**Locations (first hit wins):** `C:\HypercubeEtalon\data\` then
`C:\HypercubeWTF\data\`

**Required files** (uncompressed IDX, exact names):

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are the usual public MNIST binaries (LeCun et al.). We do not ship
them in git.

**Download example** (run from the deploy folder, or save into it):

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

On Windows, any tool that downloads those four `.gz` files and decompresses
them into one of the deploy folders is fine.

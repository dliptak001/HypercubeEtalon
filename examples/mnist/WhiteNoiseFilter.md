# The etalon preprocessor as a white-noise filter

HypercubeWTF asked a simple question. Pack an image onto the cube, dump
white noise onto that field, and ask whether a frozen preprocessor
helps the CNN head more than handing the noisy pack straight to it.
Their answer was yes. On clean digits the reservoir costs almost
nothing. Under heavy noise it holds eight or nine points that the
pack-only path loses. The write-up is
`HypercubeWTF/examples/mnist/WhiteNoiseFilter.md`.

This is the same experiment on Etalon. One etalon transit instead of a
reservoir episode. Same packing, same kind of head, same noise on the
packed field. Training stays clean. Only the test field gets noisy.

The answer is yes again. On a clean test set the two paths are almost
tied — bypass 0.979, etalon transit 0.976. Once the noise gets
serious, the transit pulls ahead and stays ahead. At σ = 0.5 the gap
is 5.2 points (0.851 vs 0.799). It peaks at 6.9 points around
σ = 0.7–0.8 and sits near 6.6 points all the way out to σ = 1.0.

That is the same story WTF told. Smaller gap, different machine. Their
reservoir is not this etalon transit, and the two tables are not one
experiment.

MNIST is just a convenient test bed. What we are scoring is a packed
field going into a HypercubeCNN, with or without the etalon transit in
front of it.

## How the comparison works

Every image is packed the same way: the full 28×28 digit in the low
addresses, a centered crop filling the rest of the 1024-vertex cube.
Both arms train on the clean 60,000-image training set. At test time
we add independent Gaussian noise of strength σ to every vertex of
the packed field — no clipping — and score the 10,000-image test set.

The only difference is what the readout sees.

- **Bypass** — the packed field, noise and all.
- **Etalon transit** — that same field after one frozen transit.

Each path trains once. Then we score the same noise ladder on both,
σ from 0 to 1 in steps of 0.1, always with noise seed `0x7E57`.

## What happened

![MNIST test noise: etalon transit vs Bypass](etalon_mnist_noise_comp.png)

On a clean field, or with only a little grain (σ = 0.1), bypass is a
few tenths of a point better. At σ = 0.2 they are tied at 0.967. From
σ = 0.3 on, the etalon transit is ahead, and the lead grows until it
levels off around six and a half points. Both paths stay well above
chance (0.10) even at σ = 1.0.

Bypass has no trouble fitting the clean training set — train accuracy
is 0.998 on both arms. It just does not recognize those digits once
the test field is full of snow. The transit puts the noisy field
somewhere the clean-trained head still knows how to read.

WTF saw the same split. At σ = 0.5 their bypass sat around 0.84–0.85
and the reservoir around 0.93. This sweep’s etalon transit at that
same noise is 0.851, about where WTF’s bypass was, and this bypass is
0.799. Clean accuracy here (0.976 / 0.979) is right next to WTF’s
clean pair (both about 0.979). Same shape. Different heights.

## The numbers

One Release run. Both arms trained to 0.998. After one etalon transit
the 1024 bank values averaged 0.1448 in size.

| sigma | transit | bypass | transit − bypass |
|------:|--------:|-------:|-----------------:|
| 0.0 | 0.976 | 0.979 | −0.003 |
| 0.1 | 0.974 | 0.977 | −0.003 |
| 0.2 | 0.967 | 0.967 | +0.000 |
| 0.3 | 0.947 | 0.934 | +0.014 |
| 0.4 | 0.910 | 0.875 | +0.035 |
| 0.5 | 0.851 | 0.799 | +0.052 |
| 0.6 | 0.784 | 0.723 | +0.062 |
| 0.7 | 0.717 | 0.648 | +0.069 |
| 0.8 | 0.649 | 0.580 | +0.069 |
| 0.9 | 0.587 | 0.521 | +0.066 |
| 1.0 | 0.535 | 0.469 | +0.066 |

## Settings

What `etalon_mnist.cpp` used for this run.

| | |
|--|--|
| Cube | dim 10, 1024 vertices |
| Etalon transit | `subcube_dim = 5` (32 steps) |
| Input / weight scale | 1.0 / 0.15 |
| Weight seed | 3458567978345987 |
| Readout | 1 layer, 16 channels, max pool, no activation |
| Epochs | 100 transit, 20 bypass |
| Learning rate | 0.003 |
| Packing | PadLowCenter |
| Train / test | 60,000 / 10,000 |
| Noise | σ = 0, 0.1, …, 1.0, seed `0x7E57` |

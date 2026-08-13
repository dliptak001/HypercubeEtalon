# Scratch notes: does the bounce shred, and does it ever win?

Not a paper. Notes from the first Exciter-vs-bypass runs after face
walks (`halvings`) landed. Release MinGW, same 1-layer pooled TANH head
unless said otherwise. Bypass = packed field straight to that head.

What we hoped to learn:

1. Does a long async bounce destroy class signal?
2. If we shorten the walk, does signal come back?
3. On clean MNIST, can the bank beat skip-the-walk?

Short answers: (1) yes, a full-cube walk can wreck an easy field.
(2) yes, shortening the walk restores most of it. (3) not on clean
PadLowCenter MNIST, even at nearly full train. The leftover gap is
small and stable (~0.6 points), not a sample-size problem.

## Synth (dim 7, N = 128)

Six made-up classes, 64 train / 32 test per class, field noise 0.22.
`in_scale = 0.30`, `wt_scale = 0.25`, 80 epochs. Bypass is always
perfect on this task. That is the point: the raw field is already
easy.

| halvings | Walk M | Exciter train / test | Bypass test | Delta |
|----------|--------|----------------------|-------------|-------|
| 0 | 128 (whole cube) | 0.714 / 0.703 | 1.000 | −0.297 |
| 1 | 64 | 0.992 / 0.979 | 1.000 | −0.021 |
| 2 | 32 | 0.997 / 0.984 | 1.000 | −0.016 |

The walk is a real knob. Full cube almost shreds a field the head
already solves. Half and quarter let the field through. Bypass still
wins. Synth cannot crown the bank; it can only show that a short bounce
is not a hash.

## MNIST (dim 10, N = 1024, PadLowCenter)

Packed 28×28 plus a 15×16 center crop. Same head family (1 conv + max
pool, TANH). `test_noise` off on every row below. Knobs below are
playground settings, not the shipped demo defaults.

### Small subset (1 000 train / 500 test)

`halvings` default at the time (half-cube), `in_scale = 0.50`,
`wt_scale = 0.20`, 40 epochs.

| Path | Train | Test | Wall |
|------|-------|------|------|
| Exciter | 0.921 | 0.812 | ~62 s |
| Bypass | 0.988 | 0.844 | ~2 s |
| Delta | | −0.032 | |

An earlier scale guess (`in = 0.08`, `wt = 0.05`) collapsed the bank
to chance (0.10 / 0.10). Too-small weights are a shredder. The 0.50 /
0.20 pair is “in the game.”

### Medium subset (10 000 / 5 000)

`halvings = 3` (M = 128), `in_scale = 0.50`, `wt_scale = 0.20`.

| Path | Train | Test | Wall |
|------|-------|------|------|
| Exciter | 0.995 | 0.929 | 95 s |
| Bypass | 0.994 | 0.935 | 26 s |
| Delta | | −0.006 | |

### Informal pair

One later playground run sat at **0.923 / 0.935** (Exciter / bypass).
Same shape: near tie, skip-the-walk a hair ahead.

### Nearly full train (58 862 / 5 000)

`TakePerClass` cannot invent extra digits, so this is “all the train
file will give,” not a fake 60 000. `halvings = 6` (M = 16),
`in_scale = 0.20`, `wt_scale = 0.20`, `lr_max = 0.002`, 40 epochs.
`test_noise=off`. First-map `mean|y| = 0.044` (small, still structured).

| Path | Train | Test | Wall |
|------|-------|------|------|
| Exciter | 0.996 | **0.965** | 185 s (collect+train 182, test 3) |
| Bypass | 0.998 | **0.972** | 162 s |
| Delta | | **−0.006** | |

Same −0.6 points as the 10k run. More data lifted both (~0.93 → ~0.97)
and did not close the gap. Collect at M = 16 is almost as cheap as
bypass. The bounce is a whisper and still does not match the raw pack.

## Reading

- A long bounce can destroy useful information. That fear was fair.
- A short bounce does not. Signal survives. That is the hope.
- On **clean** packed MNIST, the packed field is the better feature
  for this head. The leftover gap is a few tenths, not a recipe win.
- Under **test-field AWGN** the bank does earn its keep. A later
  60k / 5k sweep (`halvings = 6`) crossed bypass between σ = 0.1 and
  0.2 and held about +8 to +9 points from σ = 0.5 to 1.0. Same
  relative shape as the HypercubeWTF white-noise pre-filter, lower
  absolute scores. Write-up:
  [`examples/mnist/WhiteNoiseFilter.md`](examples/mnist/WhiteNoiseFilter.md).

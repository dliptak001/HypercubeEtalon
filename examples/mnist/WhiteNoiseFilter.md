# Exciter as a white-noise pre-filter for HypercubeCNN

Scratch study, not a paper. One Release `etalon_mnist` sweep after the
example grew a train-once / score-many test-noise table. The question
is the same one HypercubeWTF asked in
`HypercubeWTF/examples/mnist/WhiteNoiseFilter.md`:
does a **frozen** cube preprocessor help the HypercubeCNN head when the
packed field is hit with additive white Gaussian noise?

Short answer: **yes, in the same relative way.** Clean data is a near
tie (bypass a hair ahead). Under strong test AWGN the Exciter holds
about eight to nine points that bypass loses. That is the WTF headline
shape. It is **not** a claim that Exciter matches WTF’s absolute
accuracy, or that a reflection is an orbit.

MNIST is only the evaluation vehicle. The pipeline under test is pack →
optional Exciter bank → HypercubeCNN on a length-N field.

## The A/B

Both arms start from the same packed field. The only change is what
the head sees:

```text
Bypass   — packed field (no walk)
Exciter  — bank after the frozen XOR F/B reflection
```

Train stays clean. Test noise is i.i.d. N(0, σ) on every packed vertex
after pack, no clamp, seed_base `0x7E57`. Same grid on both paths after
each path trains once.

## The claim

**A short Exciter walk is nearly transparent on clean packed MNIST and
is a white-noise pre-filter under strong test AWGN, in the same
relative sense as the HypercubeWTF reservoir episode.**

![MNIST test noise: Exciter vs Bypass](etalon_mnist_noise_comp.png)

| Condition | Bypass (pack → readout) | Exciter (pack → walk → readout) |
|-----------|-------------------------|----------------------------------|
| Clean or mild AWGN (σ ≤ 0.1) | Strong (0.965–0.968) | Matchable (0.962–0.966) |
| Strong AWGN (σ = 0.5) | 0.712 | 0.797 |
| Peak logged gap | | +0.090 at σ = 0.6 |

The gap at σ = 0.5 is **+8.4 points**. WTF’s multi-seed σ = 0.5 gap was
**+8.2 to +8.7 points** (bypass ≈0.84–0.85, reservoir ≈0.93). Same
delta family, different absolute floor.

## Logged sweep

One run. Train 60000 / test 5000, dim 10, `subcube_dim = 4` (M = 16),
`in_scale = 0.2`, `wt_scale = 0.2`, PadLowCenter. Exciter 100 epochs,
bypass 40. Noise seed `0x7E57`.

| sigma | exciter | bypass | delta (exciter − bypass) |
|------:|--------:|-------:|-------------------------:|
| 0.0 | 0.966 | 0.968 | −0.002 |
| 0.1 | 0.962 | 0.965 | −0.003 |
| 0.2 | 0.949 | 0.944 | +0.005 |
| 0.3 | 0.925 | 0.889 | +0.036 |
| 0.4 | 0.867 | 0.802 | +0.064 |
| 0.5 | 0.797 | 0.712 | +0.084 |
| 0.6 | 0.724 | 0.634 | +0.090 |
| 0.7 | 0.647 | 0.562 | +0.085 |
| 0.8 | 0.579 | 0.494 | +0.086 |
| 0.9 | 0.521 | 0.442 | +0.079 |
| 1.0 | 0.475 | 0.396 | +0.079 |

Operating picture, same as WTF:

- **Clean / light noise** — preprocessor is optional. Bypass can sit a
  few tenths ahead.
- **Crossover** — between σ = 0.1 and 0.2.
- **Heavy white noise** — the bank is the difference. The gap opens
  through σ = 0.6 and then plateaus near +8 points out to σ = 1.0.
  Both arms stay well above chance (0.10).

Bypass still fits the clean training field, then fails on the noisy
test field. The walk reduces that mismatch: noisy packs land closer to
a region the clean-trained head still understands.

## Same property, different machine

| | HypercubeWTF (logged) | This Exciter sweep |
|--|----------------------|--------------------|
| Frozen preprocessor | Reservoir episode, T = 20 | Bank reflection, M = 16 (`subcube_dim = 4`) |
| Head sees | End-of-episode state | Length-N bank after the walk |
| Pack / dim | PadLowCenter, dim 10 | Same |
| Train | Clean 60k | Clean 60k |
| Test | 10k + field AWGN | 5k + field AWGN |
| Headline Δ at σ = 0.5 | +8.2 to +8.7 pp | +8.4 pp |
| Clean | Both ≈0.979 | 0.966 vs 0.968 |
| Abs. acc at σ = 0.5 | ≈0.93 vs ≈0.85 | 0.797 vs 0.712 |

Do **not** read the two tables as one campaign. WTF used a larger test
set, a different head activation (none vs TANH here), a different
frozen machine, and several noise seeds at σ = 0.5. The thing that
repeats is the **filter shape**: near-zero tax when the field is
clean, a multi-point lift once the field is white.

That is why “similar noise reduction properties” is the right sentence,
and “the Exciter matches the WTF reservoir” is the wrong one.

## What this does not claim

- A denoise theorem, or a win against Gaussian / Wiener / BM3D.
- Robustness to blur, occlusion, adversarial, or non-white noise.
- That every `subcube_dim` / scale pair is a pre-filter. This is one
  short-walk recipe.
- That the bank replaces a good pack on clean data. Clean, it does not.
- That Etalon replaces WTF. No orbit length T, no delay line, no
  episode IC. One map is one bank of reflections.

## Appendix — recipe for the table above

Example knobs in `etalon_mnist.cpp` at the time of the run. Not a
product default promise.

| Meaning | Where | Value |
|---------|-------|-------|
| Cube dim / N | `exciter.dim` | 10 / 1024 |
| Walk | `exciter.subcube_dim` | 4 (M = 16) |
| Mixer | `input_scaling` / `weight_scaling` | 0.2 / 0.2 |
| Weight seed | `exciter.seed` | 38715376369942979 |
| Head | `readout.*` | 1 layer, 16 ch, max pool, TANH |
| Epochs | `kExciterEpochs` / `kBypassEpochs` | 100 / 40 |
| `lr_max` | readout | 0.002 |
| Pack | `kPack` | PadLowCenter |
| Train / test | `kTrainSamples` / `kTestSamples` | 60000 / 5000 |
| Sweep | `kTestNoiseStart` / `End` / `Step` | 0 / 1 / 0.1 |
| Noise seed | `kTestNoiseSeedBase` | 0x7E57 |

Replay: `kRunBypass = true`, `kTestNoiseSweep = true`, Release build.

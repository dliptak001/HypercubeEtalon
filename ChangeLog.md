# HypercubeEtalon - Change Log

## v1.0.2 (Aug 22, 2026)
- Compare the Raman baseline extraction against the HypercubeCascade and HypercubeWTF siblings: three-host results table and training profiles in the example README, and the root and Python READMEs.

## v1.0.1 (Aug 22, 2026)
- Mirror the root README into the Python README, show the MNIST noise comparison in the White noise filter sections, and refactor the Raman baseline extraction section to match HypercubeCascade.
- Refit the Raman baseline extractor at 60 readout epochs: training RMSE 4.705, validation RMSE 4.767 on the LCOHard split, and refresh both READMEs with the new run.
- Point the Raman extraction demo at validation spectra 1971–1974 and regenerate the baseline overlay plot.
- Add HypercubeCascade to the root README ecosystem banner.
- Correct the empty-set metric note in the Python SDK doc.
- Drop the unverified bank-magnitude sentence from WhiteNoiseFilter.md and the chance line from the MNIST noise comparison chart.

## v1.0.0 (Aug 19, 2026)
- First public release. Set the CMake project version to 1.0.0.
- Add the `Etalon` product façade over `Exciter` and `Readout`: collect, train, predict, save, and load from one class.
- Add `Exciter`, a frozen hypercube reflection map with full-star gathers, tanh activation, face walks, `subcube_dim` in place of halvings, `weight_scaling`, and a dim cap of 12.
- Add `Readout`, a wrapper over the vendored HypercubeCNN 1.0.3 with cosine learning-rate decay, best-epoch restore, and a shuffled holdout.
- Fan `CollectBatch` across a persistent thread pool.
- Add Python bindings, the `hypercube-etalon` package, tests, the Python SDK guide, and a wheels workflow.
- Add the C++ SDK guide and bring the `Etalon`, `Exciter`, and `Readout` header comments in line with the live façade.
- Add the RamanBaselineExtraction example: load, normalize to [-1, 1], train, score denormalized RMSE, save and reload readout weights, and an opt-in exciter bypass. Best 30-epoch LCOHard result: validation RMSE 4.885.
- Add the Raman extract driver and `plot_extracted.py` overlay, with training and validation baseline extract figures.
- Add the MNIST example with an exciter-vs-bypass test-noise sweep and the WhiteNoiseFilter write-up.
- Add the synthetic Exciter-vs-bypass demo; Exciter is the default and bypass is opt-in.
- Rewrite the project README as a walkthrough of the etalon, the exciter, and the readout, and the examples README around the three programs.

# HypercubeEtalon

HypercubeEtalon processes spatial data of the kind presented to a CNN.
It is built from three core classes.

The **Etalon** class wraps the other two and manages training and
prediction.

The other two form a pipeline: preprocessor → readout.

The **Exciter** class is a preprocessing stage that consumes input
patterns, mixes them nonlinearly, and returns a field with the same
dimensions as the input.

The **Readout** class is a small HypercubeCNN that classifies or
regresses that field.

This is not reservoir computing.

The point of this experiment is to see if a preprocessing stage in
front of HypercubeCNN outperforms HypercubeCNN by itself. HypercubeWTF
has the same goal; it just does it a slightly different way, using a
**reservoir** with synthetic time, whereas here the preprocessor is an
**etalon**. The aim is a hypercube preprocessor effective enough that
the readout can be a single layer with a single convolutional channel
(weird, I know) and no pooling. Then training is fast, the memory
footprint is small, and little to no architectural engineering is
required for the CNN.

## Raman baseline extraction

The first real-world test is Raman: recover the slow fluorescence
under sharp molecular peaks, and do not lift the baseline into the
bands. Polynomials, asymmetric least squares, and ordinary
convolutional nets lose that argument under a dense peak cluster.
The Etalon does not.

One readout layer, one convolutional channel, no pooling, 100 epochs
on 10,000 synthetic LiCoO₂ (lithium cobalt oxide, LCO) training spectra.
Validation RMSE is 4.59 on 1,000 held-out spectra. Grey is the
spectrum, red is the ground-truth baseline, blue is the extract. Blue
stays on the fluorescence through the 480–530 stacks on both the
training overlay and the held-out validation overlay.

![Training extract, spectra 3811 through 3814](examples/RamanBaselineExtraction/extracted_baselines_training.png)

![Held-out validation extract, spectra 581 through 584](examples/RamanBaselineExtraction/extracted_baselines_validation.png)

The run log is
[`etalon_raman_ep100_LCO.txt`](examples/RamanBaselineExtraction/etalon_raman_ep100_LCO.txt).
How to reproduce the overlay, and the rest of the task write-up, live
in [`examples/RamanBaselineExtraction/`](examples/RamanBaselineExtraction/README.md).

## The Etalon

I now suspect that the hypercube will someday be recognized as the most
natural (least contrived) and at the same time the most powerful neural
network substrate that can possibly be realized.

The **Etalon** construct is just another example of how incredibly
elegant solutions can be built on that substrate.

Etalon is a term borrowed from optics. The physical etalon is a pair of
plane-parallel, highly reflective surfaces (mirrors) between which an
optical signal propagates. It is used for laser resonators,
interferometric measurement, and filtering.

On the hypercube, an `etalon` is a pair of vertices: any vertex and its
antipode. A hypercube has `N` vertices and therefore `N` etalons on the
full cube. There are far more than `N` once the cube is subdivided into
subcubes, each of which carries its own set of etalons.

The HypercubeEtalon design treats a vertex and its antipode as a
reflective cavity. All vertices in between contribute to the evolution
of an input signal. The walk goes something like this.

    LOOP:

        Pick a vertex r and its antipode r'. This defines an etalon.

        Copy the input field onto the cube. That overlay is the initial
        condition, and it is the same for every etalon.

        Starting at r, form the weighted sum of its nearest neighbors
        and write tanh of that sum into r.

        Move to the next vertex along the etalon, form the neighbor
        sum, and write tanh of that sum into that vertex.

        Order is causal: a later vertex sees values the earlier ones
        just wrote.

        Continue until the walk reaches the antipode r', then turn
        around and walk back to the starting vertex.

        The starting vertex is then updated for the second time. That
        is its final value, which is copied to an output buffer.

        For that etalon the task is done.

    GOTO LOOP

The loop repeats until every vertex (every etalon) has been processed,
which fully populates the output buffer.

Runnable programs live under [`examples/`](examples/README.md).

The Raman spectra themselves (about 1 GB) are not in this repository.

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

## The Etalon

I suspect that the hypercube will someday be recognized as the most natural (least contrived) and at the same time the most powerful neural network substrate that can possibly be realized.

The **Etalon** construct is just another example of how incredibly elegant solutions can be built on that substrate.

Etalon is a term borrow from the field of optics.  The physical etalon is a pair of plane parallel highly reflective surfaces (mirrors),
between which an optical signal propagates and is used for laser resonators, interferometric measurement, and filtering.

On the hypercube, the `etalon` is defined by a pair of vertices.  That pair consists of a vertice, any vertice, and its antipode.
On any given hypercube there are `N` vertices and therefore there are `N` possible `etalon` realizations on that hypercube.
In fact, there are far more than `N` since the full hypercube can be subdivided into many subcubes, each of which can represent a set of `etalons`.

The HypercubeEtalon design treats a vertice and its antipode as a reflective cavity.  All vertices in between contribute to the evolultion of an input signal.
How, well it goes something like this.

    LOOP:
    
        Pick a vertex r and its antipode r'.  This defines an etalon.

        Copy the input field onto the cube.  This overlay is the initial condition and it ensures a consistent initial condition for all etalons.

        Then, starting at r, compute the weighted sum with its nearest neighbors, and writes tanh of that sum into r.

        Then, incrementing the integer address of r by one, move to the next vertice, compute the weighted sum of its nearest neighbors, and write tanh of that sum into the vertice state.

        Order is causal: a later vertex sees values the earlier ones just wrote.

        Continue incrementing the address and updating the vertice states until the address reaches the antipode r, and then decrement the integer address and continue the walk all the way back down to the starting vertice.

        At that point the starting vertex is updated for the second time.  This is its final value, which is then copied to an output buffer.

        For that etalon the task is done.

    GOTO LOOP

The loop repeats until all vertices (all etalons) have been processed, which in turn fully populates the output buffer.

## Readout

The Readout is a thin HypercubeCNN wrapper so this repository can train
a head. A different train loop should call HypercubeCNN itself.

Runnable programs live under [`examples/`](examples/README.md).

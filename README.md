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

So in a sense, a vertice and its antipode form a reflective cavity.  All vertices in between contribute to the evolultion of an input signal.
How, well it goes something like this.

Pick a starting vertex `r` and its antipode `r'`.  

Copy the input field onto the cube.  This initializes the vertices - all of them. 

Then, starting at `r`, compute the weighted sum with its nearest neighbors, and writes `tanh` of that
sum into `r`.

Then, incrementing the integer address of `r` by one, move to the next vertice, compute the weighted sum of its nearest neighbors, and write `tanh` of that
sum into the vertice state.

Continue incrementing the address and updating the vertice states until the address reaches the antipode `r'`.

The walk is a **reflection**. `r` and its antipode — every bit flipped —
are the two reflectors. The walk leaves `r`, travels to the antipode,
and returns to `r`. At each vertex it reads the neighbors, forms a
weighted sum with the fixed edge weights, and writes `tanh` of that
sum in place. Order is causal: a later vertex sees values the earlier
ones just wrote. The antipodal reflector is written on the way out and
not written again on the way in.

When you get back to `r`, keep `state[r]`. That is the sample for this
start. One reflection is one pair of reflectors. The map is every
start.

## A bank of them

Do that once for every `r`. Reload the same input each time — otherwise
start 7 inherits the wreckage of start 6. Scale the input once, up
front, by `input_scaling`.

The result is N numbers: the value left at each starting vertex. That
vector is what the head sees. Same length as the field that went in.
Same addressing.

The mixer has no learned input matrix. The field on the vertices *is*
the initial condition. A reflection is sensitive to what you packed,
which is why every start reloads it.

## Smaller walks

A reflection does not need the antipode of the *whole* cube. A 6-face
of an 8-cube is still a cube. So is a half-cube, or a 4-face.

Set `exciter.subcube_dim` to the face dimension (1 … dim). Each
reflection then walks only `M = 2^subcube_dim` vertices: the face
through `r` whose low `subcube_dim` bits are free. Address that face as
`v = 0 … M−1` with physical vertex `v xor r`, so `v = 0` is `r` and
`v = M−1` is the antipodal reflector of that face. Reflect there. Come
back `v = M−2 … 0`.

One sample is still written per `r`, so the feature vector is still
length N. `subcube_dim = dim` is the whole cube; `dim-1` is a
half-cube. The default is `6` on the default dim-8 cube (`M = 64`).

Starts that share the same high bits walk the same face from different
reflectors. Cheaper walks, not fewer samples.

A short walk does not go blind to the rest of the cube. The gather is a
**full star**: every edge counts, including edges that step off the
face onto vertices this walk never updates. Those sites stay at the
scaled input — a frozen wall. A face that misses the packed picture
still sees it through that wall.

## Using it

Once the map is in hand, the rest is ordinary supervised learning.
`Etalon` is the front door: frozen Exciter, trainable head. Collect
fields, fit the head, predict.

`Exciter` and `Readout` work without `Etalon`. `ExciteCube` scales its
buffer in place and returns a pointer that dies on the next call.
`Train` continues from the current weights; for files, prefer
`SaveHcnnModel` over the unversioned `Weights` blob.

The Readout is a thin HypercubeCNN wrapper so this repository can train
a head. A different train loop should call HypercubeCNN itself.

Runnable programs live under [`examples/`](examples/README.md).

# HypercubeEtalon

HypercubeEtalon is designed to process spatial data such as you would present to a CNN, and is built from three core classes.

The **Etalon** class is a wrapper for the other two and conveniently manages training and prediction.

The other two form a pipeline:  preprocessor -> readout

The **Exciter** class is a preprocessing stage that consumes input patterns, performs nonlinear mixing, and returns a field with the same dimensions of the input.

The **Readout** class is a small HypercubeCNN that classifies or regresses that field.

This is not reservoir computing.

## The cube

That bit-flip rule is the entire adjacency relation. There is no
matrix to store and no graph to learn. Each vertex holds `dim` fixed
random weights, one per outgoing edge, drawn once from U(−1, 1) and
scaled by `weight_scaling`. Those weights never learn. They are the
etalon: the reference the rest of the experiment is measured against.

## A reflection (the etalon)

Pick a starting vertex `r`. Copy the input field onto the cube. Then
walk.

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

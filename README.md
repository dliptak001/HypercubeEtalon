# HypercubeEtalon

A field on a hypercube is already a complete object: one scalar at each
corner, neighbors by a single-bit flip. The **Exciter** is a frozen walk
on that field. From every starting corner it leaves home, reflects at
the far corner, and returns, writing `tanh` of a weighted neighbor sum
in place as it goes. The home values, taken together, are another field
of the same length. A small **HypercubeCNN**, sitting on the same graph,
learns to classify or regress that field.

You use it as a frozen front end on anything you can sit on those
corners. Pack the observation once — a spectrum one bin per corner, an
image packed onto the cube, a field you constructed — run the bank, and
train the head on what comes back. Collect, train, predict. The walk
returns a field of the same length and the same addressing, so the head
can emit a class label or another field on the same cube.

It earns its keep on spatial problems that already fit the cube:
classify a packed image, regress another field on the same corners, or
put a frozen mixer in front of the head and measure what the walk did.
A short bounce is nearly transparent on clean packed digits and is a
white-noise pre-filter when the test field is dirty. On a
spectrum-to-baseline regression the walk is mixing the head does not
have to learn. The control is always the same head on the raw field.

This is not reservoir computing.

## The cube

There are N corners, one for each bit-string of length `dim`. Label them
`0 … N−1`. Two corners share an edge exactly when their labels differ by
one bit. Flip bit `j` and you have taken a step.

That is the entire adjacency relation. There is no matrix to store and
no graph to learn. Each corner holds `dim` fixed random weights, one per
outgoing edge, drawn once from U(−1, 1) and scaled by `weight_scaling`.
Those weights never learn. They are the etalon: the reference the rest
of the experiment is measured against.

## A bounce

Pick a starting corner `r`. Copy the input field onto the cube. Then
walk.

The walk is a **bounce**. It leaves `r`, travels to the far corner —
the antipode, every bit flipped — and comes home. At each corner it
reads the neighbors, forms a weighted sum with the frozen edge weights,
and writes `tanh` of that sum in place. Order is causal: a later corner
sees values the earlier ones just wrote. The far corner is written on
the way out and not written again on the way in.

When you arrive home, keep `state[r]`. That is the sample for this
start. One bounce is a probe from one door. The map is the bank of all
of them.

## A bank of them

Do that once for every `r`. Reload the same input each time — otherwise
start 7 inherits the wreckage of start 6. Scale the input once, up
front, by `input_scaling`.

You get back N numbers: one home value per start. That vector is what
the head sees. Same length as the field you put in. Same addressing.

The mixer has no learned input matrix. The field on the corners *is*
the initial condition. A bounce is sensitive to what you packed, which
is why every start reloads it.

## Smaller walks

A bounce does not need the far corner of the *whole* cube. A 6-face of
an 8-cube is still a cube. So is a half-cube, or a 4-face.

Set `exciter.subcube_dim` to the face dimension (1 … dim). Each bounce
then walks only `M = 2^subcube_dim` corners: the face through `r` whose
low `subcube_dim` bits are free. Address that face as `v = 0 … M−1`
with physical corner `v xor r`, so `v = 0` is `r` and `v = M−1` is the
far corner of the face. Reflect there. Come back `v = M−2 … 0`.

You still write one sample per `r`, so the feature vector is still
length N. `subcube_dim = dim` is the whole cube; `dim-1` is a
half-cube. The default is `6` on the default dim-8 cube (`M = 64`).

Starts that share the same high bits walk the same face from different
doors. Cheaper walks, not fewer samples.

A short walk does not go blind to the rest of the cube. The gather is a
**full star**: every edge counts, including edges that step off the
face onto corners this walk never updates. Those sites stay at the
scaled input — a frozen wall. A face that misses the packed picture
still sees it through that wall.

## Using it

Once the map is in hand, the rest is ordinary supervised learning.
`Etalon` is the front door: frozen Exciter, trainable head. Collect
fields, fit the head, predict.

You can use `Exciter` and `Readout` without `Etalon`. `ExciteCube`
scales its buffer in place and returns a pointer that dies on the next
call. `Train` continues from the current weights; for files, prefer
`SaveHcnnModel` over the unversioned `Weights` blob.

The Readout is a thin HypercubeCNN wrapper so this repository can train
a head. If you want a different train loop, call HypercubeCNN itself.

Runnable programs live under [`examples/`](examples/README.md).

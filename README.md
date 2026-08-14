# HypercubeEtalon

You put a number on every corner of a cube. The **Exciter** walks that field
and writes back a new number for each starting corner. A small
**HypercubeCNN** head then learns to classify (or regress) those numbers.

That is the whole product. One map, one trainable head. No reservoir orbit,
no delay line, no leak.

## The cube

There are N corners, `N = 2ᵈⁱᵐ`. Label them `0 … N−1`. Two corners are
neighbors if their labels differ by a single bit. Flip bit `j` and you
step along one edge.

Each corner keeps `dim` fixed random weights, one per edge, drawn once at
construction from U(−1, 1) and scaled by `weight_scaling`. Those weights
never learn. Only the head does.


## A bounce

Pick a starting corner `r`. Copy the input field onto the cube. Then walk.

The walk is a **bounce**. It leaves `r`, hits the far corner — the
antipode, every bit flipped — and comes back to `r`. At each corner it
looks at the neighbors, takes a weighted sum, and writes `tanh` of that
sum in place. Order matters: a later corner sees values the earlier ones
just wrote.

The walk uses a **subcube** of dimension `subcube_dim`: `v = 0 … M−1`
with `M = 2^subcube_dim` and physical corner `v xor r` (so `v = 0` is
`r`, and `v = M−1` is the far corner of that face). Turn around there.
Come back `v = M−2 … 0`. Do not write that far corner a second time on
the way in. Set `subcube_dim = dim` if you want the whole cube
(`M = N`).

When you arrive home, keep `state[r]`. That is the sample for this start.

## A bank of them

Do that once for every `r`. Reload the same input each time — otherwise
start 7 would inherit the wreckage of start 6. Scale the input once, up
front, by `input_scaling`.

You get back N numbers: one bounce per corner. That vector is what the
head sees.

This is not a time-stepping reservoir. There is no leak, no spectral
radius, no learned input matrix. It is a spatial mixer, and it is
sensitive to whatever you put on the corners. That is the point of
reloading.

## Smaller walks

A bounce does not need the far corner of the *whole* cube. A 6-face of
an 8-cube is still a cube. So is a half-cube, or a 4-face.

Set `exciter.subcube_dim` to the face dimension (1 … dim) and each
bounce walks only `M = 2^subcube_dim` corners: the face through `r`
whose low `subcube_dim` bits are free. It still starts at `r`,
reflects at the far corner **of that face**, and comes home. You still
write one sample per `r`, so the feature vector is still length N.
`subcube_dim = dim` is the whole cube; `dim-1` is a half-cube. Default
is `6` on the default dim-8 cube (`M = 64`).

Starts that share the same high bits walk the same face from different
doors. Cheaper walks, not fewer samples.

The gather is a **full star**. Every edge counts, including edges that
step off the face onto corners this walk never updates. Those sites stay
at the original input — a frozen wall. A face that misses the packed
picture still sees it through that wall.


## Using it

`Etalon` is the front door: frozen Exciter, trainable head.

You can use `Exciter` and `Readout` without `Etalon`. `ExciteCube` scales
its buffer in place and returns a pointer that dies on the next call.
`Train` continues from the current weights; for files, prefer
`SaveHcnnModel` over the unversioned `Weights` blob.

The Readout is a thin HypercubeCNN wrapper so this repo can train a head.


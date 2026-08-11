# HypercubeEtalon

Experiment: map a length-N field on the Boolean hypercube through a bank of
XOR-rotated nonlinear sweeps, and collect one scalar per rotation.

## Idea

Vertices are `0 .. N-1` with `N = 2^dim`. Two vertices are neighbors if their
labels differ by one bit. Each vertex owns `dim` fixed random weights, one per
axis, drawn once at construction (`U(-1,1) * weight_scaling`).

The object that owns those weights and runs the passes is an **Exciter**.
Given an input field `x` of length N, `ExciteCube(x)` does the following for
every rotation index `r = 0 .. N-1`:

1. Copy the field onto the vertices: `state = input_scaling * x`.
2. Run a forward then reverse in-place pass over the cube. At each vertex the
   update is a weighted sum of hypercube neighbors, then `tan`. The visit order
   is the standard index order XOR-translated by `r`, so each `r` is the same
   local rule seen from a different group translate on the cube.
3. Write `output[r] = state[r]` after that pair of passes.

The result is a length-N vector: one excitation sample per rotation.

The dynamics are spatial, order-dependent (async updates), and **highly
initial-condition dependent**: the forward/reverse sweep is a strong mixing of
whatever sits on the vertices at the start of the pass. That is why each
rotation reloads `state` from `x` before running — every `r` must see the same
IC, not the wreckage of the previous rotation. There is no delay line, no
spectral-radius retune, no learned input matrix, and no leak; this is not a
time-stepping reservoir in the usual ESN sense.

## API sketch

```text
ExciterConfig cfg;          // dim, seed, input_scaling, weight_scaling
auto exc = Exciter::Create(cfg);
const float* y = exc->ExciteCube(x);   // x, y length N = exc->Size()
```

`x` must remain valid for the call and must not alias the exciter's internal
buffers. The returned pointer is into the exciter and is invalidated by the
next `ExciteCube` or destruction.

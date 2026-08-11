# HypercubeEtalon

Experiment: map a length-N field on the Boolean hypercube through a bank of
XOR-rotated nonlinear sweeps, and collect one scalar per rotation.

Vendored **HypercubeCNN** (Apache-2.0 core) lives under
`third_party/HypercubeCNN/` — see that tree’s `VENDORED.md`. The host CMake
builds `HypercubeCNNCore` and links it into the `HypercubeEtalon` target.

## Idea

Vertices are `0 .. N-1` with `N = 2^dim`. Two vertices are neighbors if their
labels differ by one bit. Each vertex owns `dim` fixed random weights, one per
axis, drawn once at construction (`U(-1,1) * weight_scaling`).

The object that owns those weights and runs the passes is an **Exciter**.
Given an input field `x` of length N, `ExciteCube(x)` does the following for
every rotation index `r = 0 .. N-1`:

1. Scale the field once in place: `x *= input_scaling`.
2. For each rotation `r`, copy that scaled field onto the vertices
   (`state = x`), then run a forward then reverse in-place pass. At each
   vertex the update is a weighted sum over the full hypercube neighbor star
   (all `dim` axes), then `tanh`. Visit order is the standard index order
   XOR-translated by `r` (same local rule from each group translate).

   - **Forward:** `v = 0 .. N-1` (last write is the logical antipode `v = N-1`,
     physical site `(N-1) xor r`).
   - **Reverse:** `v = N-2 .. 0` — turnaround after the forward endpoint; do
     not re-update `v = N-1` on entry into the return leg.

3. Write `output[r] = state[r]` after that pair of passes (value at the frame
   origin after the return leg).

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
ExciterConfig cfg;          // dim in [4, 10], seed, input_scaling, weight_scaling
auto exc = Exciter::Create(cfg);
const float* y = exc->ExciteCube(x);   // x, y length N = exc->Size()
```

### Contracts

- `x` non-null, length exactly `N = 2^dim`.
- `x` is scaled in place; a second call on the same buffer scales again.
- `x` must not alias the exciter's internal buffers.
- Returned pointer is into the exciter; invalid after the next `ExciteCube` or
  destruction.

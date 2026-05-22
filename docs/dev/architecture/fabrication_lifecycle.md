# Fabrication Lifecycle

This document records the current object lifecycle rules.

## Split of responsibilities

NeuroX separates four phases:

1. **`__init__`** Bind config, design parameters, the stable runtime context (`dtype`, `T__K`), and the per-instance fabrication shape (`inst_shape` at leaf modules, `w_logical_shape` at macros, `w_layout_shape` at xbars). Children are constructed here with their derived shapes, and `nominal_*` buffers are seeded.
2. **`fabricate()`** Auto-cascade resample of every owned module's static manufacturing variation. Inherited from `FabricateMixin`; subclasses override only `_sample_fabricate_mismatch`.
3. **`program(weight)`** Write the macro / xbar's owned weight state from one logical integer weight. Manually dispatched per layer (no auto-cascade).
4. **`snapshot(*, shape)`** (leaf-only) Materialise per-call dynamic noise into a frozen `*Snapshot` dataclass consumed by `solve_dc / convert / vec_mat_mul`. Never persisted on the module.

## Why this split exists

- shape-derived state is committed once at `__init__`, not learned at runtime;
- static fabrication mismatch is resamplable but is *not* a per-call event;
- runtime dynamic noise stays local to one call;
- the split keeps a future functional `program/fabricate + run` API possible.

## Canonical signatures

### Leaf circuits (analog / digital / device)

```python
def __init__(self, *, cfg, name, inst_shape, dtype, T__K) -> None: ...

def _sample_fabricate_mismatch(self) -> None: ...   # override point
def snapshot(self, *, shape: tuple[int, ...]) -> <Name>Snapshot: ...  # for devices / analog with dynamic noise
```

- `inst_shape` is the per-instance fabrication shape, committed at construction.
- The auto-cascading `fabricate()` is inherited from `FabricateMixin`.
- Subclasses override `_sample_fabricate_mismatch` only if they own static state; otherwise the inherited no-op is the correct body.
- `snapshot(*, shape)` lives only on leaves whose dynamic noise the solver consumes; the `shape` here is the per-call broadcast shape, separate from `inst_shape`.

### Xbar tiles

```python
def __init__(self, *, cfg, name, w_layout_shape, dtype, T__K) -> None: ...

def fabricate(self) -> None: ...      # auto-cascade (via FabricateMixin)
def program(self, w) -> None: ...     # write tile-native digit state
def vec_mat_mul(self, x) -> Tensor: ...
```

- `w_layout_shape` is the full digit-tensor shape the xbar's `program(w)` will receive: `(*prefix, data_num, digit_num, row_num)`. The xbar's `_inst_shape` is `w_layout_shape[:-3]`.
- `program(w)` validates `w.shape == self._w_layout_shape`, then writes through to the underlying child state (RRAM / digits buffer).
- `IdealXbar` registers a 0-d `nominal_digits` buffer and a `digits` actual buffer; `program(w)` reassigns the actual buffer.

### Macros (xbar-backed and ideal)

```python
def __init__(self, *, cfg, name, w_logical_shape, dtype, T__K, ideal_xbar) -> None: ...

def fabricate(self) -> None: ...                                              # auto-cascade
def program(self, weight) -> None: ...                                        # logical → organize → child program
def matmul(self, input, bias, mult, rshift, zp) -> Tensor: ...                # pure forward; reads programmed state
```

- `w_logical_shape` is the operator-facing weight shape, typically `(*prefix, N, K)` (linear) or `(groups, out/g, in/g·kh·kw)` (grouped conv).
- The macro derives every child's `inst_shape` symbolically from `w_logical_shape` + cfg in `__init__`, then builds children with those shapes.
- `program(weight)` runs the macro's `_organize_w` and dispatches to the xbar's `program(...)`. Digital helpers and the readout chain participate via the auto-cascade only.
- `matmul` does **not** take `weight` — it reads the previously-programmed state.

### Operators

```python
def fabricate(self) -> None: ...   # drives self.macro.fabricate()
def program(self) -> None: ...     # drives self.macro.program(self.weight_int)
def forward(self, input) -> Tensor: ...
```

`replace.fabricate_model(model)` and `replace.program_model(model)` walk every `NeuroxOperator` and drive these two entries.

## Buffer pattern

Every module that holds tensor state uses two `register_buffer(..., persistent=False)` calls at `__init__`:

- `nominal_<name>__<unit>` — the design template (usually 0-d scalar; sometimes a per-cell template tensor with no `inst_shape` expansion).
- `<name>__<unit>` (or just `<name>`) — the actual state; initialised to `nominal_*.clone()` so it is well-defined before any `fabricate()` / `program(...)` call.

`fabricate()` and `program(...)` write into the actual buffer **via attribute reassignment** (`self.X = new_tensor`). They do **not** call `register_buffer(...)` a second time; PyTorch's `__setattr__` updates the buffer slot in place while preserving the `persistent=False` flag and keeping `.to(device)` / `.to(dtype)` working.

Buffers are kept `persistent=False` so `.to(...)` migrates them but they stay out of `state_dict`. The persisted source of truth is the upper-layer `nn.Parameter` (or operator-level integer weight tensor); a `program(w)` call at load time regenerates the per-cell state.

## Three stages of physical state

Detailed in [`state_holding.md`](state_holding.md). Summary:

1. **Nominal** — design intent. Written at `__init__`. `nominal_*` 0-d (or per-cell template) buffer.
2. **Actual** — post-fabrication / post-programming. Written by `_sample_fabricate_mismatch` (mismatch) or `program(...)` (programmed weight). `*` buffer (no prefix).
3. **Snapshot** — per-call dynamic noise. Returned by `snapshot(...)` as a frozen `*Snapshot` dataclass. Never a buffer.

## Re-callability and nominal templates

- `fabricate()` may be called any number of times. Each call re-samples mismatch from the unchanged `nominal_*` buffers — no state accumulates across calls.
- `program(...)` may be called any number of times. Each call overwrites the actual weight state with a fresh encoding of the input.
- The two methods write **orthogonal state pieces**: `fabricate()` touches mismatch; `program()` touches programmed weight. The call order between them is free, but both must have been called once before `matmul` will produce meaningful (post-fabrication, post-programming) output. Until then, broadcasting against the 0-d nominal yields the deterministic, mismatch-free, zero-weight baseline.

## Typical operator-loop cadence

Training (noise-aware QAT):

```python
for step in training:
    optimizer.step()
    program_model(model)            # write the freshly-updated weights
    for forward in step:
        fabricate_model(model)      # resample mismatch
        y = model(input)            # forward consumes programmed + sampled state
```

Inference:

```python
program_model(model)                 # once at load time
fabricate_model(model)               # once at load time
loop:
    y = model(input)                 # no re-program, no re-fabricate
```

## Caveats

- **DDP / multi-GPU**: each rank samples its own mismatch realisation. If consistent mismatch across ranks is required, the user is responsible for coordinating RNG seeds before `fabricate_model(...)`. The lifecycle itself does not synchronise.
- **`torch.use_deterministic_algorithms(True)`** conflicts with the RNG paths used inside `_sample_fabricate_mismatch` and `apply_*` helpers. Disable deterministic mode (or disable the noise toggles) when using either.
- **QAT backprop**: the `weight` tensor passed to `program(...)` should be autograd-tracked (e.g., a `Parameter` produced by an STE) for gradients to flow from the macro output back to the float weight. Buffer reassignment preserves autograd; nothing in the lifecycle detaches.

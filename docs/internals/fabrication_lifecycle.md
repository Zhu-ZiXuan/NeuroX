# Fabrication Lifecycle — Implementation

## Summary

A module's state is established in four phases — `__init__` (bind config, commit shape, seed nominal buffers), `fabricate()` (resample static mismatch via the `FabricateMixin` auto-cascade), `program(...)` (write the programmed weight), `snapshot(...)` (materialise per-call dynamic noise) — feeding the run methods (`solve_dc` / `convert` / `vec_mat_mul` / `matmul`). The three state tiers these phases write are detailed in [state_holding](state_holding.md).

## Design decisions

- **Shape is committed at `__init__`, never at runtime.** `inst_shape` (leaf modules and xbars) and `w_logical_shape` (macros) are bound at construction; children are built then with their derived shapes and `nominal_*` buffers seeded. Deriving shape at first forward would couple tensor allocation to data flow and defeat the construct-once / resample-many design. The full digit-tensor shape an xbar's `program(w)` expects is derived from `inst_shape + config + subclass geometry` at `__init__` and exposed as `self._w_layout_shape` for validation only.
- **`fabricate()` is one inherited auto-cascade; only `_sample_fabricate_mismatch` is overridden.** A pre-order walk (self, then each `FabricateMixin` child, with `nn.ModuleList` / `nn.ModuleDict` transparently expanded) means a new layered module participates in fabrication for free; cascade-only container nodes inherit the no-op body and own no sampling code. Rejected — each parent manually driving its children (every refactor of the tree edits every parent); chosen — structural recursion over `self.children()`.
- **`program(...)` is manually dispatched per layer, not cascaded.** Programming consumes one logical weight that only the owning macro / xbar can organize and encode, so there is no tree-uniform argument to cascade. Auto-cascading it would force a single weight shape onto unrelated children. `fabricate()` and `program(...)` write orthogonal state pieces (mismatch vs programmed weight); their call order is free.
- **Actual state is written by attribute reassignment, not a second `register_buffer`.** `fabricate()` / `program(...)` do `self.X = new_tensor`; PyTorch's `__setattr__` updates the buffer slot in place, preserving the `persistent=False` flag and keeping `.to(device)` / `.to(dtype)` working. Re-registering would churn the buffer registry and risk dropping the persistence flag.
- **Buffers are `persistent=False`; the weight is the persisted source of truth.** Per-cell state stays out of `state_dict`; the upper-layer `nn.Parameter` (or operator-level integer weight) is what persists, and a `program(w)` at load time regenerates the per-cell state. This keeps checkpoints device- and geometry-independent.

## Contracts & invariants

- **Host requirements for the cascade.** A `FabricateMixin` host must also inherit `nn.Module` (the cascade walks `self.children()`) and must set `self._inst_shape: tuple[int, ...]` in `__init__`. Subclasses override `_sample_fabricate_mismatch(self) -> None` only if they own static state. Non-`FabricateMixin` children are silently skipped, so auxiliary `nn.Module` / `nn.Parameter` attributes coexist without participating.
- **Two-buffer pattern per stateful module.** `nominal_<name>__<unit>` (design template, usually 0-d, registered `persistent=False`) and `<name>__<unit>` (actual state, initialised to `nominal_*.clone()` so it is well-defined before any `fabricate()` / `program(...)`).
- **Re-callability.** Both `fabricate()` and `program(...)` may be called any number of times; each resamples / re-encodes from the unchanged nominal template, so no state accumulates. Both must have run once before `matmul` / `vec_mat_mul` yields post-fabrication, post-programming output; until then, broadcasting against the 0-d nominal gives the deterministic, mismatch-free, zero-weight baseline.
- **`matmul` / `vec_mat_mul` read state, not weight.** The run methods take no `weight` argument — they read the state established by `program(...)`. `snapshot(*, shape)` lives only on leaves whose dynamic noise the solver consumes.
- **QAT autograd survives buffer reassignment.** The `weight` passed to `program(...)` should be autograd-tracked (e.g. a `Parameter` produced by an STE); buffer reassignment preserves autograd and nothing in the lifecycle detaches, so gradients flow from the macro output back to the float weight.

## Performance & resources

- The cascade is a single pre-order traversal of the module tree per `fabricate()` call — cost is linear in node count, and resampling reads the (small, often 0-d) nominal buffers, so a fabricate call's allocation scales with `inst_shape`, not with any per-call batch. The typical operator cadence — `program` once per weight update, `fabricate` per forward in noise-aware QAT, neither in inference — keeps resampling off the inference hot path.

## Gotchas

- **`torch.use_deterministic_algorithms(True)` conflicts with fabrication RNG.** The paths in `_sample_fabricate_mismatch` and the `apply_*` noise helpers use nondeterministic RNG kernels; disable deterministic mode (or the noise toggles) when fabricating.
- **DDP / multi-GPU samples per rank.** Each rank draws its own mismatch realisation; the lifecycle does not synchronise. If cross-rank-consistent mismatch is required, coordinate RNG seeds before driving `fabricate()`.
- **Forgetting either call yields the silent baseline, not an error.** Skipping `fabricate()` or `program(...)` does not raise — it falls back to the zero-weight, mismatch-free baseline via 0-d broadcast. A surprisingly trivial output usually means one of the two calls was missed.
- **Do not add a second `register_buffer` to update state.** Reassign the attribute; re-registering defeats the in-place buffer-slot update and can drop `persistent=False`.

## Known limitations

- The core library ships no model-rewrite / `replace` package: walking a model to drive per-layer `fabricate()` / `program(...)` and wrapping an `XbarMacro` as a stock-layer replacement lives application-side (illustrated by `example/lenet/quant.py`, `example/bert/quant.py`). The core public surface stops at `neurox.macro`.

## Canonical signatures

Appendix (not part of the 6-section template): per-family `__init__` shape and the override / dispatch points each family exposes.

### Leaf circuits (analog / digital / device)

```python
def __init__(self, *, config, name, inst_shape, dtype, T__K) -> None: ...

def _sample_fabricate_mismatch(self) -> None: ...   # override point
def snapshot(self, *, shape: tuple[int, ...]) -> <Name>Snapshot: ...  # for devices / analog with dynamic noise
```

- `inst_shape` is the per-instance fabrication shape, committed at construction.
- The auto-cascading `fabricate()` is inherited from `FabricateMixin`.
- Subclasses override `_sample_fabricate_mismatch` only if they own static state; otherwise the inherited no-op is the correct body.
- `snapshot(*, shape)` lives only on leaves whose dynamic noise the solver consumes; the `shape` here is the per-call broadcast shape, separate from `inst_shape`.

### Xbar tiles

```python
def __init__(self, *, config, name, inst_shape, dtype, T__K) -> None: ...

def fabricate(self) -> None: ...      # auto-cascade (via FabricateMixin)
def program(self, w) -> None: ...     # write tile-native digit state
def vec_mat_mul(self, x, *, adc_operation_point) -> Tensor: ...
```

- `inst_shape` is the per-instance multiplicity prefix. The full digit-tensor shape received by `program(w)` is `(*inst_shape, col_num, w_digit_count, row_num)`, derived inside the xbar from `inst_shape + config + subclass geometry` and exposed as `self._w_layout_shape` for validation only.
- `program(w)` validates `w.shape == self._w_layout_shape`, then writes through to the underlying child state (RRAM / digits buffer).
- `IdealXbar` registers a 0-d `nominal_digits` buffer and a `digits` actual buffer; `program(w)` reassigns the actual buffer.

### Macros (XbarMacro family)

```python
def __init__(self, *, config, name, w_logical_shape, dtype, T__K, ideal_xbar) -> None: ...

def fabricate(self) -> None: ...                                              # auto-cascade
def program(self, weight) -> None: ...                                        # logical -> organize -> child program
def matmul(self, input, *, adc_operation_point) -> Tensor: ...                # pure int matmul; matches torch.matmul
```

- `w_logical_shape` is the operator-facing weight shape, typically `(*prefix, N, K)` (linear) or `(groups, out/g, in/g * kh * kw)` (grouped conv).
- Xbar-using subclasses derive every child's `inst_shape` symbolically from `w_logical_shape` + config in `__init__`, then build children with those shapes (the inherited `_build_xbar(xbar_config=..., inst_shape=...)` helper constructs the tile).
- `program(weight)` runs the macro's `_organize_w` and dispatches to the xbar's `program(...)`. Digital helpers and the readout chain participate via the auto-cascade only.
- `matmul` does **not** take `weight` — it reads the state established by `program(...)`.
- The degenerate `IdealXbarMacro` accepts the same signature for API uniformity but has no xbar / no organize step; `program(weight)` writes the integer weight directly into `self.weight`.

### User-side operators (out of core)

Modules that wrap an `XbarMacro` into a stock-PyTorch-layer replacement live in the user's repository or in `example/`; the core public surface stops at `neurox.macro`. The conventional shape — `fabricate(self)` drives `self.macro.fabricate()`, `program(self)` drives `self.macro.program(self.weight_int)`, and `forward(input)` runs the int matmul pipeline — is illustrated by `example/lenet/quant.py` and `example/bert/quant.py`. Whatever code walks the model to drive these per-layer calls (the "model rewrite" step) also lives application-side; the core library does not ship a `replace` package.

---

- **Reference**: N/A — cross-cutting lifecycle, no reference twin.
- **Implementation**: `neurox/common/mixin/fabricate.py`, `neurox/common/circuit.py`
- **Tests**: `tests/test_signal_chain.py`, `tests/test_xbar_macro.py`
- **Decisions**: [ADR-0001](../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)

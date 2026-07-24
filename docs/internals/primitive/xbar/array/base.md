# Xbar array base

`XbarArray` defines the shape-independent array contract and returns an `XbarArraySteadyState`. Boundary clamps and their reference taps are per-call dependencies rather than owned children.

## Design decisions

- **The array is where static PPA aggregates.** A [cell](../cell/base.md) and its devices self-account no static PPA, so the array — the first level that is a full electrical circuit, cell grid plus wire infrastructure — is the reporting node: the whole grid's silicon area and static leakage ride the per-instance fields on its `XbarArrayConfig`.
- **Boundary drivers and reference taps are injected, not held.** `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V)` takes two structural `ClampDriver` roles and two scalar reference taps. The array constructs and registers none of them.
- **The array takes the analog WL drive, not an integer code.** `solve_array` receives `v_wl` with shape `[..., row_num]` and performs no code-to-voltage conversion.
- **`_sample_fabricate_mismatch` is a documented no-op.** The array owns no static state of its own — cell mismatch is sampled through the `FabricateMixin` cascade into the cell children. The base therefore overrides `_sample_fabricate_mismatch` with an empty body so `fabricate()` neither re-raises the abstract hook nor perturbs the once-per-node resample; it is a pure container in the mismatch cascade.
- **Type-parameterised so the typed surface survives subclassing.** `XbarArray` is `ModuleBase[ConfigT, PolicyT]` with `ConfigT` bound to `XbarArrayConfig` and `PolicyT` to the empty `XbarArrayPolicy` marker, and its `solve_array` is generic over the two boundary-clamp snap types (`BLSnapT` / `SLSnapT`, bound to `ClampSnap`), so a concrete array binds each var to its own subclasses with no LSP-narrowing override.

## Contracts & invariants

- **`weight_grid_shape` property.** An init-determined `tuple[int, ...]` containing the conductance-grid shape `(*inst, col, row)`. It is a property because the shape is fixed at construction.
- **`program(w_state_idx)`.** Writes the cells from one state-index tensor whose shape matches the array's own `(*prefix, col_num, row_num)` layout; the abstract method every concrete array implements.
- **`solve_array` returns an `XbarArraySteadyState`, not a DCOP.** The frozen result carries the per-column bit-line port current `i_bl_port__uA` and bit-line clamp voltage `v_bl_clamp__V`, both at full leading. Energy and latency are emitted internally through the `ProfileMixin` hooks, not returned; the array builds no DCOP of its own.
- **Boundary blocks arrive per call.** `bl_driver` / `sl_driver` are the structural `ClampDriver` role and `bl_v_ref__V` / `sl_v_ref__V` are 0-d scalar reference taps. The array constructs none of them and re-registers no child driver state; each concrete array snapshots the passed-in drivers at the call shape.
- **Abstract method set.** `weight_grid_shape`, `program`, and `solve_array` are `@abstractmethod`; `_sample_fabricate_mismatch` is the base's only concrete body (the no-op above). A concrete array supplies the three abstract members plus its own construction of the cell grid, wire buffers, and solver.

---

- **Reference**: [array family](../../../../reference/primitive/xbar/array/family.md)
- **Implementation**: `neurox/primitive/xbar/array/base.py`

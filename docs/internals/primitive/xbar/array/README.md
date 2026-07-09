# Xbar array base

The pure-array family: the abstract `XbarArray` plus the `XbarArraySteadyState` result its solve returns. The base is the shape-independent contract every concrete array (a scheme's [cell](../cell/README.md)-grid + wire + [solver](../solver.md)) satisfies. It owns no boundary driver, DAC, reference, or readout — those stay outside as peer blocks, injected into `solve_array` per call.

## Design decisions

- **`XbarArray` is a `CircuitBase`, not a bare `nn.Module`.** Unlike the [cell](../cell/README.md) — a no-PPA `nn.Module` — the array is a full electrical circuit: it is where the device children's silicon area and static leakage aggregate, so it carries the `CircuitBase` PPA surface (`area_per_inst__um2` / `leakage_per_inst__uW` via `self.config`). As a `CircuitBase` it also composes `FabricateMixin` (fab-cascade hook) and `ProfileMixin` (energy / latency emission).
- **Boundary drivers and reference taps are injected, not held.** `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V)` takes the two boundary clamp drivers (structural `ClampDriver` role) and their pre-snapshotted reference taps as call parameters and constructs no driver, DAC, or reference of its own. This dependency injection is what demotes the array to a pure array — the WL DAC, BL clamp, SL drive, and boundary reference are peers under the scheme macro.
- **The array takes the analog WL drive, not the integer code.** `solve_array` receives the analog word-line drive `v_wl` (`[..., row_num]`); the DAC conversion and its drive-noise / per-VMM accounting sit upstream in a peer block.
- **`_sample_fabricate_mismatch` is a documented no-op.** The array owns no static state of its own — cell mismatch is sampled through the `FabricateMixin` cascade into the cell children. The base therefore overrides `_sample_fabricate_mismatch` with an empty body so `fabricate()` neither re-raises the abstract hook nor perturbs the once-per-node resample; it is a pure container in the mismatch cascade.
- **Type-parameterised so the typed surface survives subclassing.** `XbarArray` is `Generic[ConfigT]` (bound to `CircuitConfig`) and its `solve_array` is generic over the two boundary-clamp snap types (`BLSnapT` / `SLSnapT`, bound to `ClampSnap`), so a concrete array binds each var to its own subclasses with no LSP-narrowing override.

## Contracts & invariants

- **`weight_grid_shape` property.** An init-determined constant `tuple[int, ...]`, the conductance grid shape `(*inst, phys_col, row)`, so a caller can size a full-leading per-instance input. It is a property because the shape is fixed at construction.
- **`program(w_state_idx)`.** Writes the cells from one state-index tensor whose shape matches the array's own `(*prefix, phys_col_num, row_num)` layout; the abstract method every concrete array implements.
- **`solve_array` returns an `XbarArraySteadyState`, not a DCOP.** The frozen result carries the per-column bit-line port current `i_bl_port__uA` and bit-line clamp voltage `v_bl_clamp__V`, both at full leading. Energy and latency are emitted internally through the `ProfileMixin` hooks, not returned; the array builds no DCOP of its own.
- **Boundary blocks arrive per call.** `bl_driver` / `sl_driver` are the structural `ClampDriver` role and `bl_v_ref__V` / `sl_v_ref__V` are 0-d scalar reference taps. The array constructs none of them and re-registers no child driver state; each concrete array snapshots the passed-in drivers at the call shape.
- **Abstract method set.** `weight_grid_shape`, `program`, and `solve_array` are `@abstractmethod`; `_sample_fabricate_mismatch` is the base's only concrete body (the no-op above). A concrete array supplies the three abstract members plus its own construction of the cell grid, wire buffers, and solver.

---

- **Reference**: [array](../../../../reference/primitive/xbar/array/README.md)
- **Concrete array**: [1T1R array](_1t1r/array.md)
- **Implementation**: `neurox/primitive/xbar/array/base.py`

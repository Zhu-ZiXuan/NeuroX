# Readout base — Implementation

## Summary

`ReadOut` (`readout/base.py`) is the abstract base of the readout-chain family: it bundles every voltage-domain block between an array's per-column boundary output and the final ADC code into one fabricate / readout lifecycle. `ReadOutConfig` is the family base config carrying only orchestration-level energy / PPA knobs; concrete configs add their topology-specific child configs. `ReadOutPolicy` is the empty marker base policy; concrete impls declare a structured `*Policy(ReadOutPolicy)` nesting one sub-policy per owned child. The lone registered impl is the offset switch-cap / mux / differential-ADC chain (`offset_switchcap_mux_adc.py`). Spec: [reference/xbar/readout](../../../reference/xbar/readout/readout.md).

## Design decisions

- **The container does shape movement and energy aggregation only** — no signal-value arithmetic outside the leaf switch-cap kernels. It never multiplies voltages by a weight tensor, never re-derives a per-slice weighted sum or a reference baseline mathematically, and never scales the reference leg by an encoding constant; the per-slice negative leg is the reference switch-cap output broadcast via `unsqueeze` + `expand` onto the grouped lattice. Keeping all electrical math in the leaves means the container cannot silently re-implement a weighted sum or a baseline.
- **The per-VMM kernel is exposed as `readout(...)`, not `forward()` / `__call__`.** The chain is non-trainable and never participates in autograd, so the explicit method skips the `nn.Module` hook plumbing and keeps the call site explicit.
- **Family dispatch is keyed on `type(config)`.** `ReadOut.from_config(...)` forwards the full init bundle to the impl registered for the concrete config type; adding a topology adds one `@ReadOut.register_key(...)` line and touches neither the factory nor a branch. See [config_and_construction](../../config_and_construction.md).
- **The abstract policy field, concrete policy argument.** A composite that holds a `ReadOut` stores the abstract `ReadOutPolicy` field type; the caller passes the concrete impl policy (e.g. `OffsetSwitchCapMuxAdcReadOutPolicy`), whose `bl_adc` sub-policy is itself an abstract `ADCPolicy` base resolved to a concrete ADC policy by the caller.

## Contracts & invariants

- **Family-wide init signature.** Every concrete impl exposes `__init__(self, *, config, policy, name, inst_shape, dtype, T__K, slice_num, digit_weights)`. `config / policy / name / inst_shape / dtype / T__K` are the standard leaf-init bundle; `inst_shape` is `(*prefix, group_num)`. `slice_num: int` is the number of slices per reference group, fixing the signal-leg switch-cap's bank-axis size. `digit_weights: tuple[float, ...]` is the per-digit positional weight vector (length `digit_count`), driving the signal-leg switch-cap's `cap_weights`. Both are structural facts of the consuming xbar / macro and are committed at construction; `digit_weights` stays a Python tuple — tensor construction happens inside the leaf switch-cap. `from_config(...)` forwards all of these through to the registered impl.
- **Grouped-lattice convention.** The chain operates on a lattice keyed by reference-group structure: `*prefix` (physical-instance prefix), `*runtime` (runtime tensor prefix — broadcast of `*prefix` with any leading activation batch dims), `group_num` (reference groups), `slice_num` (slices per group, init-time constant), and `digit_count` (digits per slice, init-time constant, $\operatorname{len}(\texttt{digit\_weights})$).
- **Public kernel signature and grouped tensor shapes.** `readout(v_signal_grouped, v_ref_grouped, *, adc_operation_point)` returns the per-slice ADC code, with `v_signal_grouped: (*runtime, group_num, slice_num, digit_count)` and `v_ref_grouped: (*runtime, group_num)`. The reference leg is expanded across the per-group slice axis directly; the chain performs no second logical lookup or geometric weighting outside the leaf switch-cap banks.
- **Construction-time sub-shape derivation.** The container owns `inst_shape = (*prefix, group_num)` and derives each child's `inst_shape` at `__init__`, committing it when the child is constructed:

  | Child | `inst_shape` | Constructed via |
  |---|---|---|
  | `signal_switchcap` | `(*prefix, group_num, slice_num)` | `SwitchCap(..., cap_weights=digit_weights)` — one cap per digit |
  | `ref_switchcap` | `(*prefix, group_num)` | `SwitchCap(..., cap_weights=(1.0,))` — single unit cap, no positional weighting |
  | `analog_mux` | `(*prefix, group_num, 1)` | `AnalogMux(...)` — direct |
  | `bl_adc` | `(*prefix, group_num, 1)` | `ADC.from_config(...)` — ADC-registry dispatch |

- **Shape commitment and auto-cascade.** Per-instance shape is committed once at `__init__`; the family inherits `FabricateMixin`, so `fabricate()` is the argument-free auto-cascade trigger that fans into the children at their already-bound shapes — there are no per-call shape arguments anywhere. The container registers no electrical state of its own.

## Performance & resources

N/A — the readout is not on the memory- or compile-critical path; the compile-sensitive work is the SAR-ADC loop reached through `bl_adc` (see [analog/adc/mcs_sar](../../analog/adc/mcs_sar.md)).

## Gotchas

- **Init-bundle pass-through is captured by the subclass.** The base `__init__` discards `policy / dtype / T__K / slice_num / digit_weights` (`del ...`) after recording `inst_shape`; the concrete impl is what consumes them to size and construct the four children. A new family member must accept the full bundle even when it ignores a member.

## Known limitations

- **No structural check that the policy tree matches the config tree.** The mirror between a concrete `*Policy` nesting (`signal_switchcap`, `ref_switchcap`, `analog_mux`, `bl_adc`) and the config composition is a hand-maintained convention; a mismatched concrete ADC policy passed for the polymorphic `bl_adc` is caught only at construction / type-check time.

---

- **Reference**: [readout](../../../reference/xbar/readout/readout.md)
- **Implementation**: `neurox/xbar/readout/base.py`, `neurox/xbar/readout/offset_switchcap_mux_adc.py`
- **Tests**: `tests/test_readout_log_gating.py`, `tests/test_xbar_physics.py`, `tests/test_xbar_adc_sampling.py`
- **Decisions**: [ADR-0001](../../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)

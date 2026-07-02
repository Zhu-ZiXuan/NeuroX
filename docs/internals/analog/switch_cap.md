# Switch cap

## Summary

`SwitchCap` (`switch_cap.py`) is the bottom-plate-sampled charge-share capacitor bank: a fixed set of weighted unit caps that passively average their sampled voltages. Spec: [reference/analog/switch_cap](../../reference/analog/switch_cap.md).

## Design decisions

- **`cap_weights` is an `__init__` argument, not a config field.** The per-cap weights are a structural deployment property (the digit-encoding the bank is committed to) - they do not change across re-fabricates - whereas the config describes the unit cell (capacitance, mismatch sigma, PPA). Putting `cap_weights` and `inst_shape` together in `__init__` matches the canonical leaf signature shared with MOSFET / ADC / DAC / mux / driver / TIA, which makes `_sample_fabricate_mismatch` a pure shape-only resampling step. Rejected - a config field: it would conflate "what cell" with "how deployed" and force the config to vary per call site.
- **The class is encoding-agnostic.** The semantic meaning of the weights (binary, unit, positional digit) is not part of the SwitchCap contract; the composing readout assigns it. This keeps one bank reusable for both the per-digit signal accumulation and the single-cap reference baseline.
- **`T__K` is an `__init__` argument, not a config field.** It sets the kT/C noise sigma - an operating-state quantity, not a design parameter - so it is bound with the operating point at construction, not frozen into the design config.

## Contracts & invariants

- **`cap_weights: tuple[float, ...]`**, length `n_caps`, all positive; tensor construction happens once inside `__init__` (the nominal cap-array buffer `nominal_c__fF = c_unit__fF * cap_weights`). Callers pass a Python tuple, not a tensor.
- **Fabricate reassigns, not re-registers.** `_sample_fabricate_mismatch` clone-expands `nominal_c__fF` to `(*inst_shape, n_caps)` and applies Pelgrom-scaled static mismatch (gated by `policy.cap_mismatch`) by attribute reassignment, not a fresh `register_buffer`.
- **Two policy switches.** `cap_mismatch` (static, at fabricate) and `sampling_thermal_noise` (dynamic kT/C, at sample) are independent.

## Performance & resources

N/A - the charge-share kernel is a per-call reduction off the memory- and compile-critical path.

## Gotchas

- **Do not treat `cap_weights` as runtime state.** It is fixed at construction; a bank committed to a digit-encoding keeps its weight template across re-fabricates. Re-deploying with different weights means a new instance, not a re-fabricate.

## Known limitations

- N/A.

---

- **Reference**: [switch_cap](../../reference/analog/switch_cap.md)
- **Implementation**: `neurox/analog/switch_cap.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.

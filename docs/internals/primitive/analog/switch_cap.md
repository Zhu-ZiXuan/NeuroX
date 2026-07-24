# Switch cap

## Design decisions

- **`cap_weights` is an `__init__` argument, not a config field.** The per-cap weights are a structural deployment property fixed at construction - they do not change across re-fabricates - whereas the config describes the unit cell (capacitance, mismatch sigma, PPA). Putting `cap_weights` and `inst_shape` together in `__init__` matches the canonical leaf signature shared with MOSFET / ADC / DAC / mux / driver, which makes `_sample_fabricate_mismatch` a pure shape-only resampling step.
- **The class is encoding-agnostic.** The semantic meaning of the weights (binary, unit, positional digit) is outside the `SwitchCap` contract.
- **`T__K` is an `__init__` argument, not a config field.** It sets the kT/C noise sigma - an operating-state quantity, not a design parameter - so it is bound with the operating point at construction, not frozen into the design config.

## Contracts & invariants

- **`cap_weights: tuple[float, ...]`**, length `n_caps`, all positive; tensor construction happens once inside `__init__` — the nominal cap-array buffer `nominal_c__fF = c_unit__fF * cap_weights` and its live per-instance copy `c__fF`.
- **Fabricate reassigns, not re-registers.** `_sample_fabricate_mismatch` clone-expands `nominal_c__fF` to `(*inst_shape, n_caps)`, applies Pelgrom-scaled static mismatch (gated by `policy.cap_mismatch`), and writes the result into `c__fF` by attribute reassignment, not a fresh `register_buffer`. The mismatch draw carries a lower floor of `0.1 * c_unit__fF` clamped onto each sampled cap, so a Gaussian tail cannot drive a sampled capacitance non-positive - the per-cap kT/C sigma and the charge-share denominator both require it positive.
- **Two policy switches.** `cap_mismatch` (static, at fabricate) and `sampling_thermal_noise` (dynamic kT/C, at sample) are independent.
- **Per-call PPA tally.** `sample_and_accumulate` self-logs a dynamic energy and a latency on every call, via `_log_dynamic_energy` / `_log_latency`. The energy is the sampled-charge term (its equation is the Reference energy model), taken on the clean sampled `v_in__V`, not the kT/C-perturbed `v_hold__V`, and reduced over the cap axis. The latency is `latency_per_op__ns * serial_op_count`, where `serial_op_count` is the position-invariant `numel // inst_count` rule taken on the *output* `v_out__V`; because `n_caps` is already summed out of `v_out__V`, the divisor is just `inst_count` and the cap axis never inflates the count.

## Gotchas

- **Do not treat `cap_weights` as runtime state.** It is fixed at construction, so re-deploying with different weights means a new instance, not a re-fabricate.

---

- **Reference**: [switch_cap](../../../reference/primitive/analog/switch_cap.md)
- **Implementation**: `neurox/primitive/analog/switch_cap.py`
- **Tests**: TODO - name the guarding test

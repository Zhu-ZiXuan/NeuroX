# Switch cap

## Design decisions

- **`cap_weights` is an `__init__` argument, not a config field.** The per-cap weights are a structural deployment property fixed at construction - they do not change across re-fabricates - whereas the config describes the unit cell (capacitance, mismatch sigma, PPA). Putting `cap_weights` and `inst_shape` together in `__init__` matches the canonical leaf signature shared with MOSFET / ADC / DAC / mux / driver, which makes `_sample_fabricate_mismatch` a pure shape-only resampling step.
- **The class is encoding-agnostic.** The semantic meaning of the weights (binary, unit, positional digit) is outside the `SwitchCap` contract.
- **`T__K` is an `__init__` argument, not a config field.** It sets the kT/C noise sigma - an operating-state quantity, not a design parameter - so it is bound with the operating point at construction, not frozen into the design config.

## Contracts & invariants

- **`cap_weights: tuple[float, ...]`**, length `_cap_num`, all positive; tensor construction happens once inside `__init__` as the nominal buffer `_nominal_c__fF = c_unit__fF * cap_weights`.
- **Fabricate creates ordinary state.** `_sample_fabricate_mismatch` clone-expands `_nominal_c__fF` to `(*inst_shape, _cap_num)`, applies Pelgrom-scaled static mismatch (gated by `policy.cap_mismatch`), and assigns `_c__fF` state. The mismatch draw carries a lower floor of `0.1 * c_unit__fF` clamped onto each sampled cap, so a Gaussian tail cannot drive a sampled capacitance non-positive; the per-cap kT/C sigma and the charge-share denominator both require it positive.
- **Two policy switches.** `cap_mismatch` (static, at fabricate) and `sampling_thermal_noise` (dynamic kT/C, at sample) are independent.
- **Per-call PPA tally.** When a profiler is active, `sample_and_accumulate` computes and emits dynamic energy. The energy is the sampled-charge term (its equation is the Reference energy model), taken on the clean sampled `v_in__V`, not the kT/C-perturbed `v_hold__V`, and reduced only over the physical cap axis. The remaining tensor is emitted as the payload and the profiler reduces the rest of it.
- **The bank's own instance axes fold into the collector's sum like any other trailing axis.** The per-cap capacitance array forces `inst_shape` into the energy tensor; `sample_and_accumulate` sums the cap axis first, so the emitted payload is `[..., *inst_shape]`, and the profiler sums every axis past the caller's own leading dims — the instance block included — with nothing declared at the emission site.

## Gotchas

- **Do not treat `cap_weights` as runtime state.** It is fixed at construction, so re-deploying with different weights means a new instance, not a re-fabricate.

---

- **Reference**: [switch_cap](../../../reference/primitive/analog/switch_cap.md)
- **Implementation**: `neurox/primitive/analog/switch_cap.py`
- **Tests**: TODO - name the guarding test

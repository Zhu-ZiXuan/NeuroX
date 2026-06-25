# ADC base — Implementation

## Summary

The ADC family: the abstract `ADC` (`adc/base.py`) carrying the registry, `from_config`, the abstract `convert` / `max_bits` / `signed_range` surface, and the multi-mode support types (`adc/_multimode.py`). Concrete topologies live alongside ([general](general.md), [mcs_sar](mcs_sar.md), [sar_mono](sar_mono.md)). Spec: [reference/analog/adc/base](../../../reference/analog/adc/base.md).

## Design decisions

- **Family dispatch keyed on config type.** `ADC.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[ADCConfig], ADC]`; adding a concrete ADC only adds a registration line and a config subclass, never touches `from_config`. `ADCConfig` is an empty marker so the polymorphic field on a parent config has a base type to name.
- **Empty marker `ADCPolicy`.** The base policy carries no switch; each concrete ADC declares its own `*Policy(ADCPolicy)` with that topology's toggles. A composite that owns an ADC stores the abstract `ADCPolicy` field type and the caller passes the concrete impl - so the policy shape follows the chosen topology, not a union of all topologies.
- **The zero code is not centralised.** Mapping the raw unsigned bucket to the signed output by subtracting a zero code is left to each topology rather than a shared base helper: different ADC families could place the zero point differently (asymmetric boundaries, single-ended). The base makes no commitment; it only fixes the signed *output range* contract.
- **Per-op latency is leaf-defined, not a base contract.** There is no `latency_per_op__ns` field on `ADC` / `ADCConfig`. Fixed-latency impls carry the field on their own config; parametric SAR impls derive latency from the runtime operating point. Each `convert` emits its own latency through the profiler side channel. A base field would force a single latency shape on topologies whose latency model genuinely differs.
- **Rounding follows `self.training`, no override flag.** Stochastic-versus-deterministic rounding is the standard `nn.Module` train/eval state; there is no constructor-time override knob, so behaviour is toggled the same way as the rest of the model.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `config / policy / dtype / T__K` so the dispatcher type-checks; concrete subclasses store the rest.
- **`convert(v_pos__V, v_neg__V, *, v_refs__V, adc_operation_point)`** takes a per-call `v_refs__V: Tensor` (all injected reference taps, shape `(*inst, num_refs)`) plus the operating point as a per-call `AdcOperationPoint` (the frozen `(adc_mode, adc_bits)` pair). The ADC self-holds no reference; `adc_mode` indexes the injected tensor's trailing axis, so the legal mode bound is `0 <= adc_mode < v_refs__V.shape[-1]` (checked against the tensor, not a config field). Single-mode impls validate `adc_mode == 0` and `adc_bits == max_bits` and may ignore `v_refs__V`; multi-mode impls accept any pair inside their envelope.
- **Signed-code output, with the clamp before the shift.** Every `convert` clamps the raw bucket to `[0, n_codes-1]` *before* subtracting the topology zero code `z`, landing in `[-z, n_codes-1-z]`; the clamp must precede the subtraction because stochastic-rounding jitter (floor_bucketize or SAR LSB jitter) can push the raw bucket out of range and the subtraction would otherwise emit an out-of-range signed code. For the current linear / uniform members `n_codes = 2**adc_bits` and a symmetric zero code gives `[-2**(adc_bits-1), 2**(adc_bits-1)-1]`.
- **Calibration types.** `ADCMode` is the `(n_bits, n_states, max_signal)` record the multi-mode LUTs use; `AdcOperationPoint` is the per-call selection; `AdcCalibrationRecord` is one `(adc_mode, adc_bits) -> rescale_factor` row. Static PPA (`area_per_inst__um2`, `leakage_per_inst__uW`) comes from `CircuitBase`.

## Performance & resources

N/A at this level - the per-conversion cost is topology-specific (see [mcs_sar](mcs_sar.md) for the SAR-loop compile considerations).

## Gotchas

- **Do not clamp after the zero shift.** Subtracting the zero code from an unclamped raw bucket can produce a signed code outside the legal range; the order is fixed (clamp, then shift).
- **The signed convention is required by the consumer rescale, not cosmetic.** A positive rescale factor mapping a signed code to a signed dot product is only well-defined when the code carries the input sign directly; an ADC that returned unsigned codes would silently break the tile-boundary rescale and the calibrate tool's positivity invariant.

## Known limitations

- N/A.

---

- **Reference**: [adc base](../../../reference/analog/adc/base.md)
- **Implementation**: `neurox/analog/adc/base.py`, `neurox/analog/adc/_multimode.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.

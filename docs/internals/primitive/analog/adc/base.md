# ADC base

The ADC family: the abstract `ADC` (`adc/base.py`) carries the registry, `from_config`, the abstract `convert` / `max_bits` / `signed_range` surface, and the multi-mode support types (`adc/_multimode.py`). The ADC owns the digitize step alone; current-to-voltage clamping belongs to the tia / voltage_driver, column multiplexing to the voltage_mux, and the code-to-scale rescale to the caller. The ADC self-holds no reference - every reference tap arrives as a per-call argument.

## Design decisions

- **Family dispatch keyed on config type.** `ADC.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[ADCConfig], ADC]`; adding a concrete ADC only adds a registration line and a config subclass, never touches `from_config`. `ADCConfig` is an empty marker so the polymorphic field on a parent config has a base type to name.
- **Empty marker `ADCPolicy`.** The base policy carries no switch; each concrete ADC declares its own `*Policy(ADCPolicy)` with that topology's toggles, so the policy shape follows the chosen topology, not a union of all topologies.
- **The zero code is not centralised.** Mapping the raw unsigned bucket to the signed output by subtracting a zero code is left to each topology rather than a shared base helper: different ADC families could place the zero point differently (asymmetric boundaries, single-ended). The base makes no commitment; it only fixes the signed *output range* contract.
- **Per-op latency is leaf-defined, not a base contract.** There is no `latency_per_op__ns` field on `ADC` / `ADCConfig`. Fixed-latency impls carry the field on their own config; parametric impls derive latency from the runtime operating point. Each `convert` emits its own latency through the profiler side channel. A base field would force a single latency shape on topologies whose latency model genuinely differs.
- **Rounding follows `self.training`, no override flag.** Stochastic-versus-deterministic rounding is the standard `nn.Module` train/eval state; there is no constructor-time override knob, so behaviour is toggled the same way as the rest of the model.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only. The full signature lets `from_config` construct any impl uniformly; the base forwards `config` / `name` / `inst_shape` to `CircuitBase` (which stores `self._inst_shape` and registers profiler bookkeeping) and discards `policy` / `dtype` / `T__K`, which the subclass init captures.
- **`convert(v_pos__V, v_neg__V, *, v_refs__V, adc_operation_point)`** takes a per-call `v_refs__V: Tensor` (all injected reference taps, shape `(*inst, num_refs)`) plus the operating point as a per-call `AdcOperationPoint` (the frozen `(adc_mode, adc_bits)` pair). The ADC self-holds no reference; `adc_mode` indexes the injected tensor's trailing axis, so the legal mode bound is `0 <= adc_mode < v_refs__V.shape[-1]` (checked against the tensor, not a config field). Single-mode impls validate `adc_mode == 0` and `adc_bits == max_bits` and may ignore `v_refs__V`; multi-mode impls accept any pair inside their envelope.
- **Signed-code output, with the clamp before the shift.** Every `convert` clamps the raw bucket to `[0, n_codes-1]` *before* subtracting the topology zero code `z`, landing in `[-z, n_codes-1-z]`; the clamp must precede the subtraction because stochastic-rounding jitter can push the raw bucket out of range and the subtraction would otherwise emit an out-of-range signed code. When `n_codes = 2**adc_bits`, a symmetric zero code gives `[-2**(adc_bits-1), 2**(adc_bits-1)-1]`. The signed output is a hard contract: the code carries the input sign directly, as the downstream positive-scale recovery model requires (see Reference).
- **Calibration types.** `ADCMode` is the `(n_bits, n_states, max_signal)` record the multi-mode LUTs use; `AdcOperationPoint` is the per-call selection; `AdcCalibrationRecord` is one `(adc_mode, adc_bits) -> rescale_factor` row. Static PPA (`area_per_inst__um2`, `leakage_per_inst__uW`) comes from `CircuitBase`.

---

- **Reference**: [adc base](../../../../reference/primitive/analog/adc/family.md)
- **Implementation**: `neurox/primitive/analog/adc/base.py`, `neurox/primitive/analog/adc/_multimode.py`
- **Tests**: TODO - name the guarding test

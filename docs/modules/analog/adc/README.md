# ADC Family

This directory documents the ADC family under `neurox/analog/adc/`.

Current rules:

- `ADCConfig` is the family base config
- concrete ADCs each own a concrete config subclass
- `ADC.from_config(...)` dispatches by concrete config type and takes `inst_shape` at construction
- the family inherits `FabricateMixin`; each impl overrides `_sample_fabricate_mismatch` (auto-cascade `fabricate()` is provided by the mixin)
- runtime conversion uses explicit `(mode, bits)` per call
- stochastic-vs-deterministic rounding follows `self.training`; no override knob

Current concrete implementations:

- `GeneralADC`
- `McsSarAdc`
- `SarAdcMono`

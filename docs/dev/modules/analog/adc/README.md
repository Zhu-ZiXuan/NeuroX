# ADC Family

This directory documents the ADC family under `neurox/analog/adc/`.

Current rules:

- `ADCConfig` is the family base config
- concrete ADCs each own a concrete config subclass
- `ADC.from_config(...)` dispatches by concrete config type
- `stochastic` is a family-level explicit runtime parameter, not a config field
- runtime conversion uses explicit `(mode, bits)` per call

Current concrete implementations:

- `GeneralADC`
- `McsSarAdc`
- `SarAdcMono`

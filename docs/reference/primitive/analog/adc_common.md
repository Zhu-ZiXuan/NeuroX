# Shared ADC types

Domain-neutral types that carry no physical model — they name the multi-mode descriptor and the calibration row the architecture layer keys its ADC calibration against.

## Types

- `AdcMode` — one operating mode of a multi-mode ADC: `(bits, n_states, max_signal)`, with derived `code_num = 2 ** bits` and `lsb = max_signal / code_num`. `max_signal` is the full-scale input magnitude in the family's native analog quantity — voltage [V] for a voltage ADC, current [uA] for a current ADC.
- `AdcCalibrationRecord` — one `(mode, bits) -> rescale_factor` row of a calibration lookup. The recovery model is `M_ideal ≈ code · rescale_factor` with `rescale_factor` strictly positive; the quantize inverse is `code = floor(M_ideal / rescale_factor)`.

## Placement

`AdcMode` and `AdcCalibrationRecord` are descriptor/calibration records that no ADC sub-package imports, so they stay in `neurox/primitive/analog/adc_common.py`. The consuming architecture layer keys its calibration LUT on the `(adc_mode, adc_bits)` pair and stores the externally-calibrated `AdcCalibrationRecord` rows.

---

- **Implementation**: `neurox/primitive/analog/adc_common.py` (`AdcMode`, `AdcCalibrationRecord`)
- **Voltage family**: [voltage ADC family](voltage_adc/family.md)
- **Current family**: [current ADC family](current_adc/family.md)

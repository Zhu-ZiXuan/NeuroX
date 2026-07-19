# Shared ADC types

Domain-neutral types shared by the [voltage ADC](voltage_adc/README.md) and [current ADC](current_adc/README.md) families. They carry no physical model — they name the runtime operating point, the multi-mode descriptor, and the calibration row both families quantize against.

## Types

- `AdcOperationPoint` — the frozen `(adc_mode, adc_bits)` pair passed to `convert` per call. `adc_mode` selects the reference set the family defines — a tap of the injected reference tensor (voltage) or a threshold-ladder row (current); valid range `[0, mode count)`. `adc_bits` is the active resolution, `1 <= adc_bits <= max_bits`.
- `AdcMode` — one operating mode of a multi-mode ADC: `(n_bits, n_states, max_signal)`, with derived `n_codes = 2 ** n_bits` and `lsb = max_signal / n_codes`. `max_signal` is the full-scale input magnitude in the family's native analog quantity — voltage [V] for a voltage ADC, current [uA] for a current ADC.
- `AdcCalibrationRecord` — one `(adc_mode, adc_bits) -> rescale_factor` row of a calibration lookup. The recovery model is `M_ideal ≈ code · rescale_factor` with `rescale_factor` strictly positive; the quantize inverse is `code = floor(M_ideal / rescale_factor)`.

## Placement

All three types live in `neurox/primitive/analog/adc_common.py` so neither family depends on the other. The consuming architecture layer keys its calibration LUT on `AdcOperationPoint` and stores the externally-calibrated `AdcCalibrationRecord` rows.

---

- **Implementation**: `neurox/primitive/analog/adc_common.py`
- **Voltage family**: [voltage ADC family](voltage_adc/family.md)
- **Current family**: [current ADC family](current_adc/family.md)

# Analog circuits

Analog circuit models on the array boundary and in the readout chain: the current-to-voltage clamp, the integer-code drivers, the column transport, the charge-share bank, and the digitizers. Each document specifies a circuit's transfer characteristic, its non-idealities, and its parameters.

- [voltage_driver](voltage_driver.md) — the generic Thevenin voltage-source clamp (finite series output resistance; the ideal constant-voltage source is its zero-resistance limit).
- [voltage_mux](voltage_mux.md) — differential voltage column transport with common-mode / differential-mode noise.
- [voltage_reference](voltage_reference.md) — the multi-output voltage reference source (static PPA only; per-die tolerance + per-read noise on the taps).
- [current_mirror](current_mirror.md) — the ideal single-ended ratio current-copy block (data-dependent rail energy).
- [current_mux](current_mux.md) — the ideal single-ended N:1 time-share current-transport block (data-dependent rail energy + serial latency).
- [current_reference](current_reference.md) — the multi-output current reference source (static PPA only; per-die tolerance + per-read noise on the taps).
- [current_subtractor](current_subtractor.md) — the single-ended current subtractor (magnitude of the leg difference + a sign bit; ratio-mismatch and sign-offset nonidealities).
- [switch_cap](switch_cap.md) — the bottom-plate-sampled charge-share capacitor bank.
- [voltage_adc/](voltage_adc/README.md) — the voltage ADC family: the differential voltage-domain digitizers (boundary-bucketize and SAR topologies) emitting signed codes.
- [current_adc/](current_adc/README.md) — the current ADC family: the single-ended magnitude current digitizers emitting unsigned codes.
- [adc_common](adc_common.md) — the domain-neutral operating-point / calibration types shared by both ADC families.
- [voltage_dac/](voltage_dac/README.md) — the voltage DAC family: integer-code to analog-drive-voltage conversion.
- [current_dac/](current_dac/README.md) — the current DAC family: integer-code to single-ended analog-current conversion.
- [tia/](tia/README.md) — the TIA family: the BL clamp transimpedance amplifier.

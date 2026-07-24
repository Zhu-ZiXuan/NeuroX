# Analog

How the analog circuit layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the `AnalogBase` root, its empty config / policy markers, and where each block's per-instance PPA fields are declared.
- [voltage_driver](voltage_driver.md) — the generic Thevenin clamp, its `ClampDriver` conformance without inheritance, and why it owns no conduction energy.
- [voltage_mux](voltage_mux.md) — the differential voltage-transport leaf.
- [voltage_reference](voltage_reference.md) — the multi-output voltage reference leaf, its static-only PPA, and the fabricate-then-runtime non-ideality split.
- [current_mux](current_mux.md) — the ideal N:1 time-share current leaf, a lossless value map that emits nothing.
- [current_reference](current_reference.md) — the multi-output current reference leaf and why its bias power is static leakage, not derived from the taps.
- [switch_cap](switch_cap.md) — the charge-share bank and its `cap_weights` placement.
- [unmodeled](unmodeled.md) — the static-PPA-only seat, why it owns no functional path, and how the composing macro bills its dynamic energy through a channel.
- [voltage_adc/](voltage_adc/README.md) — the voltage ADC family: registry dispatch, signed-code helper placement, multi-mode.
- [current_adc/](current_adc/README.md) — the current ADC family: registry dispatch, single-ended magnitude / unsigned-code contract, shared-lane fabrication.
- [voltage_dac/](voltage_dac/README.md) — the voltage DAC family: registry dispatch, the unsigned code-to-voltage contract, the code-only convert with no per-call operating point.
- [current_dac/](current_dac/README.md) — the current DAC family: registry dispatch, the unsigned code-to-current contract, single-ended drive and leaf-defined per-op latency.

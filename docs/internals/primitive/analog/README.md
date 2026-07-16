# Analog

How the analog circuit layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the `AnalogBase` root, its empty config / policy markers, and where each block's per-instance PPA fields are declared.
- [voltage_driver](voltage_driver.md) — the generic Thevenin clamp, its `ClampDriver` conformance without inheritance, and why it owns no conduction energy.
- [voltage_mux](voltage_mux.md) — the differential voltage-transport leaf.
- [voltage_reference](voltage_reference.md) — the multi-output voltage reference leaf, its static-only PPA, and the fabricate-then-runtime non-ideality split.
- [current_mirror](current_mirror.md) — the ideal ratio current-copy leaf, and why it reports neither dynamic energy nor static PPA of its own.
- [current_mux](current_mux.md) — the ideal N:1 time-share current leaf, a lossless value map that emits nothing.
- [current_reference](current_reference.md) — the multi-output current reference leaf and why its bias power is static leakage, not derived from the taps.
- [current_subtractor](current_subtractor.md) — the single-ended magnitude-and-sign current-difference leaf, and why it reports neither dynamic energy nor static PPA of its own.
- [switch_cap](switch_cap.md) — the charge-share bank and its `cap_weights` placement.
- [voltage_adc/](voltage_adc/README.md) — the voltage ADC family: registry dispatch, signed-code helper placement, multi-mode.
- [current_adc/](current_adc/README.md) — the current ADC family: registry dispatch, single-ended magnitude / unsigned-code contract, shared-lane fabrication.
- [voltage_dac/](voltage_dac/README.md) — the voltage DAC family: registry dispatch, the unsigned code-to-voltage contract, the code-only convert with no per-call operating point.
- [current_dac/](current_dac/README.md) — the current DAC family: registry dispatch, the unsigned code-to-current contract, single-ended drive and leaf-defined per-op latency.
- [tia/](tia/README.md) — the TIA family: snap typing and the solver-facing surface.

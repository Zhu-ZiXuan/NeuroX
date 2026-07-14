# Analog

How the analog circuit layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the `AnalogBase` triple, and which leaves are sized (`ProfileMixin` + PPA) vs bare.
- [voltage_driver](voltage_driver.md) — the generic Thevenin clamp, its `ClampDriver` conformance without inheritance, and why it owns no conduction energy.
- [voltage_mux](voltage_mux.md) — the differential voltage-transport leaf.
- [voltage_reference](voltage_reference.md) — the multi-output voltage reference leaf, its static-only PPA, and the fabricate-then-runtime non-ideality split.
- [current_mirror](current_mirror.md) — the ideal ratio current-copy leaf, and why it is bare of energy and static PPA.
- [current_mux](current_mux.md) — the ideal N:1 time-share current leaf, a lossless value map that emits nothing.
- [current_reference](current_reference.md) — the multi-output current reference leaf and why its bias power is static leakage, not derived from the taps.
- [switch_cap](switch_cap.md) — the charge-share bank and its `cap_weights` placement.
- [adc/](adc/README.md) — the ADC family: registry dispatch, signed-code helper placement, multi-mode.
- [dac/](dac/README.md) — the DAC family container.
- [tia/](tia/README.md) — the TIA family: snap typing and the solver-facing surface.

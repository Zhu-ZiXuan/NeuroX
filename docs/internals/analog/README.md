# analog — Implementation

How the analog circuit layer is built. The spec is in [reference/analog](../../reference/analog/README.md); this side covers only what the code cannot tell you.

- [driver](driver.md) — the ideal SL clamp-driver and why it is not a TIA.
- [analog_mux](analog_mux.md) — the differential transport leaf.
- [switch_cap](switch_cap.md) — the charge-share bank and its `cap_weights` placement.
- [adc/](adc/README.md) — the ADC family: registry dispatch, signed-code helper placement, multi-mode.
- [dac/](dac/README.md) — the DAC family container.
- [tia/](tia/README.md) — the TIA family: snapshot typing and the solver-facing surface.

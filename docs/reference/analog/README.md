# Analog Circuits

Analog circuit models on the array boundary and in the readout chain: the current-to-voltage clamp, the integer-code drivers, the column transport, the charge-share bank, and the digitizers. Each document specifies a circuit's transfer characteristic, its non-idealities, and its parameters.

The layer is three leaf circuits (each a single concrete block a consuming circuit composes directly) plus three polymorphic families (an abstract contract with concrete implementations):

- [driver](driver.md) — the ideal constant-voltage SL clamp-driver.
- [analog_mux](analog_mux.md) — differential column transport with optional common-mode / differential-mode noise.
- [switch_cap](switch_cap.md) — the bottom-plate-sampled charge-share capacitor bank.
- [adc/](adc/README.md) — the ADC family: the current/voltage-domain digitizers (boundary-bucketize and SAR topologies).
- [dac/](dac/README.md) — the DAC family: integer-code to analog-voltage conversion.
- [tia/](tia/README.md) — the TIA family: the BL clamp transimpedance amplifier.

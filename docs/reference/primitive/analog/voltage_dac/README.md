# Voltage DAC family

The voltage DAC family converts integer codes to analog drive voltages. An abstract family contract plus its concrete implementations.

- [base](family.md) — the abstract `VoltageDac` contract: the unsigned code domain and the code-to-voltage map.
- [general](general.md) — the LUT-based voltage DAC with additive drive-thermal noise.

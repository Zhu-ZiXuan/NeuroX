# Current DAC family

The current DAC family converts integer codes to single-ended analog output currents. An abstract family contract plus its concrete implementations.

- [base](family.md) — the abstract `CurrentDac` contract: the unsigned code domain and the code-to-current map.
- [general](general.md) — the LUT-based current DAC with additive output noise.

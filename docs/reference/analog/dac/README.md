# DAC Family

The DAC family converts integer codes to analog voltages - in the crossbar it drives the word lines from the integer activation code. An abstract family contract plus its concrete implementations.

- [base](base.md) — the abstract `DAC` contract: the code-to-voltage map and the nominal-operating-point LUT.
- [general](general.md) — the LUT-based DAC with optional drive-thermal noise.

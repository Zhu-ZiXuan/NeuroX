# TIA Family

The TIA family provides the bit-line (BL) clamp transimpedance amplifier: the BL-side boundary actor that holds the BL at a virtual-ground reference and absorbs the column's port current. An abstract family contract plus its concrete implementations.

- [base](base.md) — the abstract `TIA` contract: the clamp transfer function and the virtual-ground reference.
- [general](general.md) — the linear Thevenin-input plus resistive-transimpedance concrete TIA.
- [opamp_tia](opamp_tia.md) — the op-amp plus NMOS-pseudo-resistor concrete TIA.

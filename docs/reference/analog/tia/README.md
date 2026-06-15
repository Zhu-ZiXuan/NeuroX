# TIA Family

The TIA family provides the bit-line (BL) clamp transimpedance amplifier: the BL-side boundary actor that holds the BL at a virtual-ground reference and absorbs the column's port current. An abstract family contract plus its concrete implementations.

- [base](base.md) — the abstract `TIA` contract: the clamp transfer function and the virtual-ground reference.
- [opamp_tia](opamp_tia.md) — the op-amp plus NMOS-pseudo-resistor concrete TIA.

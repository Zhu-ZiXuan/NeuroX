# CimUnit family

How the `CimUnit` family is built.

- [base](base.md) — `CimUnit` registry root, the `_build_cim_macro` helper, the `chunk_pad_along` primitive, and the construction scaffolding shared by xbar-using modes.
- [direct](direct.md) — `DirectCimUnit`: the no-slice pipeline.
- [inter_array_slice](inter_array_slice.md) — `InterArraySliceCimUnit`: cross-plane `Sw` layout and its aggregate dual.
- [intra_array_slice](intra_array_slice.md) — `IntraArraySliceCimUnit`: intra-tile `Sw` fold and its aggregate dual.
- [ideal](ideal.md) — `IdealCimUnit`: the degenerate member that joins the registry without a tile.
- [slicer/](slicer/README.md) — the `Slicer` ABC observable surface and the two value-decomposition implementations.

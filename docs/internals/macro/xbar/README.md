# XbarMacro — Implementation

How the `XbarMacro` registry family is built: the abstract registry root and the per-mode organize/aggregate shape pipelines. Spec: [reference/macro/xbar](../../../reference/macro/xbar/README.md).

- [base](base.md) — `XbarMacro` registry root, the `_build_xbar` helper, the `chunk_pad_along` primitive, and the construction scaffolding shared by xbar-using modes.
- [direct](direct.md) — `DirectXbarMacro`: the no-slice pipeline.
- [inter_array_slice](inter_array_slice.md) — `InterArraySliceXbarMacro`: cross-plane `Sw` layout and its aggregate dual.
- [intra_array_slice](intra_array_slice.md) — `IntraArraySliceXbarMacro`: intra-tile `Sw` fold and its aggregate dual.
- [ideal](ideal.md) — `IdealXbarMacro`: the degenerate member that joins the registry without a tile.

# CimEngine family

How the `CimEngine` family is built.

- [base](base.md) — `CimEngine` registry root, the shared backend (`_init_engine_backend`, the sub-phase plane machinery `_unroll_sub_phase`, the hoisted value-range / ADC surface and `program` skeleton), the `_build_cim_macro` helper, the `chunk_pad_along` primitive, and the `[Sa, Sw, Tc, Tr]` layout convention.
- [direct](direct.md) — `DirectCimEngine`: the no-slice pipeline with the identity `DirectSlicer` on the activation path.
- [inter_array_slice](inter_array_slice.md) — `InterArraySliceCimEngine`: cross-plane `Sw` layout and its aggregate dual.
- [intra_array_slice](intra_array_slice.md) — `IntraArraySliceCimEngine`: intra-tile `Sw` fold and its aggregate dual.

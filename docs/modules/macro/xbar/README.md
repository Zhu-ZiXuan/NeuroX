# XbarMacro Family

This subdirectory documents the `XbarMacro` registry family.

## Files

- [`base.md`](base.md) — `XbarMacro` abstract registry root and `XbarMacroConfig`.
- [`direct.md`](direct.md) — `DirectXbarMacro` (`Sw = Sa = 1`, transcode-only, no slicers).
- [`inter_array_slice.md`](inter_array_slice.md) — `InterArraySliceXbarMacro` (Strategy 1: `Sw` across xbar planes).
- [`intra_array_slice.md`](intra_array_slice.md) — `IntraArraySliceXbarMacro` (Strategy 2: `Sw` gathered within one xbar).
- [`ideal.md`](ideal.md) — `IdealXbarMacro` (degenerate; no xbar, pure integer matmul reference).

See [`docs/dev/architecture/xbar_macro.md`](../../../dev/architecture/xbar_macro.md) for the architectural rules (slice → organize → tile, organize ↔ aggregate duality, `Sw` layout vs `Sa` schedule).

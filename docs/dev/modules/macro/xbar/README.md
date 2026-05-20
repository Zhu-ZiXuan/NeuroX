# Xbar-Backed Macros

This subdirectory documents the xbar-backed macro family.

## Files

- [`base.md`](base.md) — `XbarMacro` abstract class and `XbarMacroConfig`.
- [`inter_xbar_slice.md`](inter_xbar_slice.md) — `InterXbarSliceMacro` (Strategy 1: `Sw` across xbar planes).
- [`intra_xbar_slice.md`](intra_xbar_slice.md) — `IntraXbarSliceMacro` (Strategy 2: `Sw` gathered within one xbar).

See [`docs/dev/architecture/xbar_macro.md`](../../../architecture/xbar_macro.md) for the architectural rules (slice → organize → tile, organize ↔ aggregate duality, `Sw` layout vs `Sa` schedule).

# XSliceStage

`XSliceStage` pairs call-time input decomposition with the inverse `Sx`
digital aggregation. It does not participate in geometric placement.

## Implementations

`DirectXSliceStage` uses `DirectSlicer`, appends a structural `Sx=1` axis, and
removes that axis without a digital operation.

`SerialXSliceStage` uses `SerialSlicer`. Its radix is the number of integer
values in one macro input interval,
`R_a = x_hi - x_lo + 1`. The stage-owned shift adder reduces `Sx` after the
geometric and weight-slice reductions. Its physical multiplicity is `(G,)`.

## Contracts

- `slice` maps `[...,M,K]` to `[...,M,K,Sx]`.
- `aggregate` receives `[...,M,Sx,G,Q]` and returns `[...,M,G,Q]`.
- `value_range` is sourced from the owned slicer and is the engine's published
  logical input range.
- `Sx` is a serial time axis, never a macro instance axis. The canonical macro
  weight layout carries `Sx=1`; runtime input broadcasting materializes the
  serial reads.

---

- **Reference**: [input slicing](../../../../../reference/architecture/unit/cim/engine/x_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/x_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`

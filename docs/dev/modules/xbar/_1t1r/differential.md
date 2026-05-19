# `neurox/xbar/_1t1r/differential.py`

## Current status

Placeholder — currently empty.

`differential.py` is reserved for the differential array-organization scheme: positive / negative column pairs encode signed weights, in contrast to the offset-coded single sub-array plus reference columns used by `offset.py`. When implemented, this file will hold the column-layout indices, the digit encoding policy, and the array orchestration for the differential scheme.

## Relationship to the physical core

Differential and offset are parallel array-organization schemes; both stand on the same encoding-agnostic physical core abstraction. The current concrete core is `CircuitCore1T1R` (the 1T1R cell topology), but the array scheme itself is decoupled from the cell topology — future core variants slot in without changes to the array-organization files.

See also:

- `offset.md`
- `circuit_core.md`
- `../../../architecture/mapping.md`

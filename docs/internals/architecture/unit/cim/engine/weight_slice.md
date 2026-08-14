# WeightSliceStage

`WeightSliceStage` pairs program-time weight decomposition and physical layout
with the inverse `Sw` digital aggregation.

## Implementations

### Direct

`DirectWeightSliceStage` uses `DirectSlicer`, reports the macro's weight range,
and retains a structural `Sw=1` macro plane. It does not instantiate a shift
adder.

### Inter-plane

`InterWeightSliceStage` uses `SimpleSlicer`. Every `Sw` slice occupies a
separate macro plane, so:

- `layout_geometry(output_num)` returns `(Q=output_num, Sw)`.
- Weight layout is `[...,Sw,Tc,G,D,L,output_num]`.
- The macro instance count contains the `Sw` factor.
- The stage-owned shift adder reduces `Sw` after `Tc`.

Its shift-adder multiplicity is `(G,)`; the output-port work is
represented by runtime tensor elements.

### Intra-port

`IntraWeightSliceStage` maps adjacent `Sw` slices to adjacent output ports in
one macro:

- `Q = floor(output_num / Sw)`.
- Weight layout flattens `(Q,Sw)` into the physical output axis and pads unused
  trailing ports.
- Physical macro-plane count remains one.
- Aggregation trims idle ports, unflattens `(Q,Sw)`, and shift-adds `Sw`.

`Sw > output_num` is invalid because no complete logical weight fits in one
macro. Non-divisible output capacity is valid and leaves trailing ports idle.

## Contracts

- `slice` always appends `Sw`, including size-one direct operation.
- `arrange_weight` consumes the canonical placement tensor
  `[...,D,G,Q,Tc,L,Sw]`.
- `aggregate` consumes `[...,Sx,Sw,G,output_num]` after `P` and `Tc` have
  already been reduced and returns `[...,Sx,G,Q]`.
- `value_range` is computed by the owned slicer from the macro-level weight
  range, slice count, radix, and encoding.

---

- **Reference**: [weight slicing](../../../../../reference/architecture/unit/cim/engine/weight_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/weight_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`

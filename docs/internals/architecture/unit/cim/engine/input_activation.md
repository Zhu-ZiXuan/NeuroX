# InputActivationStage

`InputActivationStage` adapts a generic active-input limit to the CIM `P`
axis and owns the digital reduction that removes that axis.

## Planning and masking

`make_input_activation_plan` partitions a local input block of length `L`
into

$$P=\left\lceil\frac{L}{A}\right\rceil$$

groups, where `A` is `cim_macro.max_active_num`.
`make_activation_group_mask` returns `_active_input_mask[P,L]`; every local
input belongs to exactly one group. The non-persistent mask follows module
device moves and is reconstructible from runtime geometry.

`unroll_input_phases` maps `[...,L]` to `[...,P,L]` by retaining only the
group selected by each `P` index. This operation is independent of geometric
block slots and therefore occurs before `PlacementStage.unroll_block_steps`.

## Digital ownership

`phase_accumulator` reduces `P` after each macro read. Its physical
multiplicity is `(Sw, Tc, G)`, covering every physical weight
plane, contraction tile, and macro group that receives an independent code.

## Contracts

- `max_active_num` limits the number of unmasked positions, independent of
  their runtime values.
- `unroll_input_phases` preserves every axis except the inserted `P`.
- `accumulate_phases` removes only `P`.
- The stage does not know block slots, `D`, or output placement.

---

- **Reference**: [input activation](../../../../../reference/architecture/unit/cim/engine/input_activation.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/input_activation.py`
- **Tests**: `tests/architecture/unit/test_engine_input_packing.py`

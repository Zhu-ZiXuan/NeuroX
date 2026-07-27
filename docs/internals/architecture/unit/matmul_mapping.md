# Matrix-multiplication mapping

The mapping utilities separate substrate-independent planning from the
execution model that consumes it.

## Design decisions

- **Geometry has no scheduling vocabulary.** A placement plan partitions the
  contraction dimension, divides logical outputs into blocks, balances those
  blocks across groups, and assigns each block a slot inside its group. It
  does not decide whether a group or slot is realized in space or time.
- **Activation grouping is independent.** An activation plan partitions the
  local inputs of one weight block according to an active-input limit. The
  limit cannot alter weight placement.
- **Routing remains independent.** Block-slot routing and activation-group
  masking are derived separately. A concrete execution engine decides how
  and where to compose them.
- **Plans contain data only.** The immutable dataclasses own no modules,
  configuration, policy, PPA, device lifecycle, or substrate-specific state.

## Contracts & invariants

`make_matmul_placement_plan` derives:

- `contraction_block_size`: logical contraction values in one weight block;
- `contraction_partition_num`: blocks covering the contraction dimension;
- `output_block_num`: blocks covering the logical outputs;
- `block_group_capacity`: maximum blocks one group can contain;
- `block_group_num`: groups needed to contain all output blocks;
- `block_slot_num`: balanced slots used by each group.

Output block `block_slot * block_group_num + block_group` occupies tile input
positions beginning at `block_slot * contraction_block_size`. A missing final
block denotes padding.

`make_input_activation_plan` partitions local input indices into
`activation_group_num` groups of at most `active_input_limit` values.

`make_block_slot_routing` translates geometric slots into
`gather_index[block_slot,tile_input]` and
`slot_mask[block_slot,tile_input]`.
`make_activation_group_mask` independently returns
`activation_mask[activation_group,block_input]`. Neither utility introduces
an execution-order interpretation.

---

- **Reference**: [unit family](../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/matmul_mapping.py`
- **Tests**: `tests/architecture/unit/test_matmul_mapping.py`

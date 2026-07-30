# PlacementStage

`PlacementStage` adapts generic matrix-multiplication geometry to CIM macro
axes and owns the digital reduction for contraction partitions `Tc`.

## Planning

`make_matmul_placement_plan` derives immutable runtime geometry from logical
`[N,K]`, macro `input_num`, and the weight-layout block width `Q`. The generic
plan fields map to CIM terms as follows:

- `contraction_block_size` becomes `L`;
- `contraction_partition_num` becomes macro axis `Tc`;
- `block_group_num` becomes macro axis `G`;
- `block_slot_num` becomes runtime axis `D`.

Logical block `b = dG + g` occupies input slot
`[dL,(d+1)L)` in macro group `g`. This minimizes `G` before balancing the
blocks over `D` uniform steps; missing final blocks are zero-programmed.

## Block routing

`make_block_slot_routing` produces generic gather indices and slot masks,
which the stage registers as `_input_source_index[D,input_num]` and
`_block_slot_mask[D,input_num]`. Both are non-persistent buffers because they
must follow module device moves but can be reconstructed.

`unroll_block_steps` accepts input that already carries `P`, inserts `D`, and
routes every local block into its assigned macro-input slot. Caller-owned
leading axes stay left of `D`; missing weight-batch axes are inserted as
size-one broadcast axes.

## Digital ownership

- `contraction_accumulator` reduces `Tc` and has physical multiplicity
  `(w_parallel, Sw, G)`.

Here `Sw` means physical macro planes, so it is one for direct and intra-port
layouts and equals `w_slice_num` for inter-plane layout.

## Contracts

- `partition_weight` returns `[...,D,G,Q,Tc,L,Sw]`.
- `pack_weight` accepts `[...,Sw,Tc,G,D,L,output_num]`.
- `organize_x` returns `[...,M,Sa,Sw=1,Tc,G=1,L]`.
- `unroll_block_steps` accepts `[...,*inst_shape,P,L]` and returns
  `[...,D,P,*inst_shape,input_num]`.
- `restore_output` accepts `[...,D,*w_batch,M,G,Q]`, restores logical block
  order, and trims only the output padding.

---

- **Reference**: [placement](../../../../../reference/architecture/unit/cim/engine/placement.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/placement.py`
- **Tests**: `tests/architecture/unit/test_matmul_mapping.py`, `tests/architecture/unit/test_engine_input_packing.py`

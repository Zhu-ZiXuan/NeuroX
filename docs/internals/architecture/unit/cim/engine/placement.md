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
routes every local block into its assigned macro-input slot. Missing
weight-batch axes are inserted as size-one broadcast axes.

The insertion index is rank arithmetic, so it cannot tell a caller batch dim
from a weight batch dim and places the `D, P` pair left of the weight-batch
slot, giving `[*caller_leading[:execution_index], D, P, *w_batch, M, Sa, Sw, Tc, G, input_num]`.
Only the dims left of `execution_index` are therefore provably caller-owned.
Without a weight batch that index is the whole caller prefix rank, so every
caller-owned leading axis does stay left of `D`; with one, the caller prefix is
split and `D` sits inside it.

That split bounds the reporting resolution: the profiler reads the leftmost
`leading_rank` dims of an energy tensor as the caller block, so a weight-batched
engine resolves per caller unit operation only up to `execution_index`, and a
measurement declaring more attributes `D` to a caller axis. Totals are unaffected
either way — every axis past the prefix is summed regardless — so this is a
resolution bound, not a correctness one.

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

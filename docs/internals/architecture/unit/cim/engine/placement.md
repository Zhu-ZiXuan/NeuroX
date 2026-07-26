# PlacementStage

`PlacementStage` owns workload geometry, balanced short-vector packing, the
runtime input mask, and the digital reductions for the two geometric
partial-sum axes `P` and `Tc`.

## Planning

`PlacementPlan.build` derives immutable runtime geometry from logical
`[N,K]`, macro `input_num`, and the weight-layout block width `Q`:

- `L = min(K, input_num)`
- `Tc = ceil(K / L)`
- `B = ceil(N / Q)`
- `C = floor(input_num / L)`
- `G = ceil(B / C)`
- `D = ceil(B / G)`

Logical block `b = dG + g` occupies input slot
`[dL,(d+1)L)` in macro group `g`. This minimizes `G` before balancing the
blocks over `D` uniform steps; missing final blocks are zero-programmed.

## Scheduling

For macro limit `A = max_active_num`, the stage derives
`P = ceil(L / A)`. `_input_source_index[D,input_num]` routes each local input
vector into its block slot, while `_active_input_mask[D,P,input_num]` leaves
at most `A` positions selected in each phase. Both tensors are non-persistent
buffers because they must follow module device moves but can be reconstructed.

`unroll_input_schedule` inserts `[D,P]` immediately before the
instance-aligned block. Caller-owned leading axes stay left of `D`; missing
weight-batch axes are inserted as size-one broadcast axes.

## Digital ownership

- `phase_accumulator` reduces `P` and has physical multiplicity
  `(w_parallel, Sw, Tc, G)`.
- `contraction_accumulator` reduces `Tc` and has physical multiplicity
  `(w_parallel, Sw, G)`.

Here `Sw` means physical macro planes, so it is one for direct and intra-port
layouts and equals `w_slice_num` for inter-plane layout.

## Contracts

- `partition_weight` returns `[...,D,G,Q,Tc,L,Sw]`.
- `pack_weight` accepts `[...,Sw,Tc,G,D,L,output_num]`.
- `organize_x` returns `[...,M,Sa,Sw=1,Tc,G=1,L]`.
- `restore_output` accepts `[...,D,*w_batch,M,G,Q]`, restores logical block
  order, and trims only the output padding.

---

- **Reference**: [placement](../../../../../reference/architecture/unit/cim/engine/placement.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/placement.py`
- **Tests**: `tests/architecture/unit/test_engine_input_packing.py`

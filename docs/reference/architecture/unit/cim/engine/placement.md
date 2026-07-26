# Geometric placement

Placement maps the logical `N` and `K` dimensions onto macro capacity without
changing numeric precision.

It partitions `K` into `Tc` blocks of length `L`, partitions `N` into blocks
of width `Q`, packs multiple short-`K` blocks into disjoint macro input slots,
and balances those blocks over `G` physical macro groups and `D` serial steps.
The `max_active_num` limit further masks each input slot into `P` serial
phases.

The stage owns two digital operations:

- a serial accumulator over the `P` reads belonging to one block;
- an accumulator over the `Tc` contraction blocks belonging to one dot
  product.

The mapping and both accumulators are configured together through
`PlacementStageConfig`. Precision slicing is independent and is described by
[weight slicing](weight_slice.md) and [input slicing](x_slice.md).

Uniform control is intentional: zero-padded groups in the final `D` step
still execute. The simulator does not assume fine-grained per-group enables.

---

- **Internals**: [placement internals](../../../../../internals/architecture/unit/cim/engine/placement.md)
- **Engine**: [CIM engine](family.md)

# Geometric placement

Placement maps the logical `N` and `K` dimensions onto macro capacity without
changing numeric precision.

It partitions `K` into `Tc` blocks of length `L`, partitions `N` into blocks
of width `Q`, packs multiple short-`K` blocks into disjoint macro input slots,
and balances those blocks over `G` physical macro groups and `D` serial steps.

The stage owns the accumulator over the `Tc` contraction blocks belonging to
one dot product.

The geometric mapping and accumulator are configured together through
`PlacementStageConfig`. Selected-input grouping is handled independently by
[input activation](input_activation.md); precision slicing is described by
[weight slicing](weight_slice.md) and [input slicing](x_slice.md).

Uniform control is intentional: zero-padded groups in the final `D` step
still execute. The simulator does not assume fine-grained per-group enables.

---

- **Internals**: [placement internals](../../../../../internals/architecture/unit/cim/engine/placement.md)
- **Engine**: [CIM engine](family.md)

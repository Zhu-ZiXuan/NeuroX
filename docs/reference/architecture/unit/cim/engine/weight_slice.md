# Weight slicing

Weight slicing widens the engine's logical weight range beyond one macro's
weight range. `SimpleSlicer` decomposes each weight into `Sw` positional
digits with radix `Rw`; the paired shift adder reconstructs

$$W=\sum_{s=0}^{S_w-1}w_sR_w^s.$$

Three layouts are available.

| Configuration | Physical layout | Logical outputs per block |
|---|---|---|
| `DirectWeightSliceStageConfig` | no decomposition; structural `Sw=1` | `output_num` |
| `InterWeightSliceStageConfig` | one macro plane per slice | `output_num` |
| `IntraWeightSliceStageConfig` | adjacent output ports carry one weight's slices | `floor(output_num/Sw)` |

Inter-plane layout spends `Sw` times as many macro instances. Intra-port
layout keeps one macro plane but reduces useful logical output width and may
leave trailing output ports idle. Both sliced layouts own an `Sw` shift adder;
the direct layout performs no digital operation.

The stage publishes the complete logical `w_value_range` computed by its
slicer. The macro remains responsible only for the value range accepted by
one slice.

---

- **Internals**: [weight-slice internals](../../../../../internals/architecture/unit/cim/engine/weight_slice.md)
- **Engine**: [CIM engine](family.md)

# CimEngine base

`CimEngine` is the registry root for logical slicing, tiling, macro execution,
and digital aggregation. The base owns macro construction, input-phase masking,
value-range and ADC delegation, the shared `program` skeleton, and
`_chunk_pad_along`. Each variant owns its organize/aggregate pair.

## Design decisions

- **Registry dispatch keyed on config and policy types.** A variant declares `@CimEngine.register_neurox_module(config_type=..., policy_type=...)`; `from_config` passes both objects to `_lookup_neurox_module`, and `RegistryMixin` constructs the internal type-pair key. The two concrete types must identify one registered pair.
- **A `ModuleBase` container, not a plain `nn.Module`.** The fabricate cascade recurses only into `FabricateMixin` children, so the engine must be one for `fabricate()` on the unit to reach the cim_macro. It is a container: `is_profile_target = False` (no own PPA — its children self-report), `_sample_fabricate_mismatch` is a pass, and `inst_shape = ()`.
- **The engine owns input-phase serialization.** The macro publishes
  `max_active_num`; the engine expands every logical input into
  `P = ceil(input_num / max_active_num)` zero-masked phases and sums exactly
  that phase axis after conversion.
- **Backend construction is shared; execution remains variant-specific.**
  `_init_engine_backend` constructs the macro, stores tiling metadata, omits
  phases containing only contraction-axis padding, and registers the static
  `[P, input_num]` mask.
- **Value ranges are leaf-sourced, base-published.** The concrete `w_value_range` / `x_value_range` properties read the protected `_w_value_range` / `_x_value_range` attrs each leaf ctor sets once after building its transcoder/slicers (the sources differ per leaf); the ADC surface delegates to `self.cim_macro` directly.
- **`Sw` layout vs `Sa` schedule split.** `Sw` (per-weight slice count) is fixed at `program` time and held until the next `program`; `Sa` (per-activation slice count) is evaluated per `matmul` call and materialized by the simulator as a batched tensor axis. Fixed at different lifecycle points, the two cannot share a uniform slice-reduction helper, and the organize step lives on the W side only.
- **Only logical macro interfaces cross the layer boundary.** `CimEngineConfig`
  owns `input_num` and `output_num`; variants use them for tiling and pass them
  to the macro constructor. They read only value ranges, activation limits, and
  ADC metadata back from the constructed macro. No engine reads physical rows,
  columns, or digit geometry.
- **`[Sa, Sw, Tc, Tr]` is the fixed leading-axis order.** Every variant's organized weight tensor places its present slice/tile axes in this canonical order ahead of the tile-owned `(data, D, row)` trailing block. A variant that does not use an axis omits it entirely rather than padding it size-1 (the `M=1` / `Sa=1` placeholders inserted for broadcast against the activation are a separate matter). Fixing the order across variants is what lets the aggregate reductions name their axes by a stable negative index.
- **No self-compiled forward.** Each variant's `matmul` is wrapped in `@torch.no_grad()` and runs eager — the library does not self-compile the forward ([compile contracts](../../../../compile/contracts.md)). The memory-sensitive cost is the tile read inside `self.cim_macro.vec_mat_mul`, whose heavy DC solve compiles as a separate regional leaf — see [array internals](../../../../primitive/xbar/array/_1t1r/array.md).

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered variant through one call shape (`config`, `policy`, `w_logical_shape`, `dtype`, `T__K`, `ideal_macro`); narrowing or reordering the constructor arguments breaks dispatch.
- **`_unroll_input_phase` inserts P immediately left of the instance-aligned
  block.** It maps trailing `[input_num]` to
  `[..., P, *span, input_num]`; the axis is retained when its size is one.
- **`matmul` owns phase and tile accumulation.** The macro returns trailing
  `[output_num]`; `phase_accumulator` reduces `_input_phase_dim` before the
  variant-specific slice reductions and the `Tc` accumulator.
- **The `program` skeleton is shared; `_organize_w` is the variant hook.** `program` shape-gates against the bound `w_logical_shape` and hands `_organize_w(weight)` to `cim_macro.program`; only the organize bodies are leaf-specific.
- **`_build_cim_macro` honours `ideal_macro`.** When enabled, the helper applies
  `.to_ideal()` after constructing the configured macro with the final
  instance multiplicity.
- **`_chunk_pad_along` is the one shared geometric primitive.** It right-pads an axis to a multiple of `chunk_size` then unflattens it into `(num_chunks, chunk_size)`, inserting the chunk axis immediately after. `pad_value` has no default. Every variant's tiling step funnels through it so the pad/unflatten convention is identical across variants.
- **The `[Sa, Sw, Tc, Tr]` order is load-bearing for the aggregate.** The aggregate reductions address their axes by negative index against the fixed leading-axis order; a variant that reorders or inserts an axis, or assumes an omitted axis is present, silently reduces the wrong dimension, so each variant's aggregate index set is variant-specific.
- **Fabrication cascades through children.** The engine registers no physical
  state of its own.

---

- **Reference**: [engine family](../../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_engine_input_phase.py`

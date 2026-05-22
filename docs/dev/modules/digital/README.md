# Digital Modules

`neurox/digital/` holds the behavioural digital-aggregation primitives used to stitch around an integer datapath:

- `accumulator.py` — modular-arithmetic accumulator over per-cycle integer outputs.
- `adder.py` — primitive integer adder.
- `subtractor.py` — primitive integer subtractor.
- `shift_adder.py` — shift-and-add reduction for digit-radix combination.
- `requantizer.py` — fixed-point multiplier / shift / bias used to fold an externally-supplied rescale factor into an integer output grid.

Every block is a thin `nn.Module + FabricateMixin` carrying a `*Config` dataclass with the standard PPA fields (`bit_width`, `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns`, `energy_per_op__fJ`). They are deliberately behavioural — no gate-level netlist semantics today.

## Construction

Family signature: `__init__(*, cfg, name, inst_shape)`. The block records its profiler instance count from `inst_shape` at construction; no per-call sampling state lives on a digital block, so the inherited `_sample_fabricate_mismatch` default no-op is the right body.

## Composition

None of the digital blocks form a polymorphic family, so they are not part of the `RegistryDispatchMixin` infrastructure — they are leaf circuits instantiated directly by concrete class name.

## `@torch.compile` fusion

These blocks are `@torch.compile`-friendly: per-op constants fold as Inductor compile-time constants, and the per-call kernel is shape-clean. The blocks themselves only need to be functional and shape-clean; the fused kernel emerges from the compiled forward path.

See also:

- `docs/dev/roadmap.md` (gate-level netlist plans for the future digital path)

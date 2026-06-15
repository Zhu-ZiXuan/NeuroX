# Digital Modules

`neurox/digital/` holds the behavioural digital-aggregation primitives used to stitch around an integer datapath:

- `accumulator.py` — modular-arithmetic accumulator over per-cycle integer outputs.
- `adder.py` — primitive integer adder.
- `subtractor.py` — primitive integer subtractor.
- `shift_adder.py` — shift-and-add reduction for digit-radix combination.

The fixed-point multiply-shift requantize step is not modelled as a separate digital block — it lives inline at the consumer's call site.

Every block inherits `CircuitBase[<Name>Config]` (so it composes `FabricateMixin + ProfileMixin + nn.Module + Generic[ConfigT]`) and carries a `<Name>Config(CircuitConfig)` dataclass: `area_per_inst__um2` and `leakage_per_inst__uW` are inherited from `CircuitConfig`; the block adds its own `bit_width`, `energy_per_op__fJ`, and `latency_per_op__ns: float`. They are deliberately behavioural — no gate-level netlist semantics today.

## Construction

Family signature: `__init__(*, config, name, inst_shape)`. The `super().__init__(...)` call initialises `nn.Module + ProfileMixin` and binds `config + _inst_shape`; static-PPA properties (`area_per_inst__um2 / leakage_per_inst__uW`) are inherited from `CircuitBase`; per-op latency is read from the block's own config. No per-call sampling state lives on a digital block, so the inherited `_sample_fabricate_mismatch` default no-op is the right body.

## Serial-op accounting

`operate(...)` is element-wise (Adder / Subtractor) or reduce-along-dim (Accumulator / ShiftAdder). Each block computes its own `serial_op_count = max(1, y.numel() // self.inst_count)` at the end of `operate` (position-invariant numel rule; reduce-ops use the output `y`, which already excludes the reduced dim), builds the latency tensor as `torch.tensor(self.config.latency_per_op__ns * serial_op_count, ...)`, and emits via `_log_dynamic_energy(energy)` + `_log_latency(latency)` (two independent calls).

## Composition

None of the digital blocks form a polymorphic family, so they are not part of the `RegistryMixin` infrastructure — they are leaf circuits instantiated directly by concrete class name.

## `@torch.compile` fusion

These blocks are `@torch.compile`-friendly: per-op constants fold as Inductor compile-time constants, and the per-call kernel is shape-clean. The blocks themselves only need to be functional and shape-clean; the fused kernel emerges from the compiled forward path.

See also:

- `docs/dev/roadmap.md` (gate-level netlist plans for the future digital path)

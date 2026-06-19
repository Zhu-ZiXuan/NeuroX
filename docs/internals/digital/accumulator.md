# accumulator — Implementation

## Summary

`Accumulator` reduces an integer tensor along one axis into a fixed-width signed register with two's-complement modular wrap, and emits its own PPA events. Spec: [reference/digital/accumulator](../../reference/digital/accumulator.md).

## Design decisions

- **Behavioural, not gate-level.** The block models the function and a flat per-op cost, not a netlist; there is deliberately no carry-tree or per-bit timing model. The rationale is that the digital periphery is not the fidelity-critical path (the analog core is), so a netlist would buy accuracy that nothing downstream consumes. The gate-level path is a roadmap item, not a present capability.
- **Modular wrap, not saturation.** The output folds modulo $2^{w}$ rather than clamping, matching a synthesized adder tree with no saturation logic. This is a correctness-relevant choice: a consumer that needs saturation must add it, and a sum that overflows aliases silently rather than pinning to the rail.
- **Not part of a registry family.** The four digital blocks share no polymorphic dispatch surface, so they stay leaf circuits instantiated by concrete class name and are kept out of `RegistryMixin`. Adding family dispatch would impose a config-type discriminator with no caller that needs to select among them at runtime.

## Contracts & invariants

- **`operate(x, dim)` reduces exactly one axis.** The reduced axis is gone from the output, so the serial-op divisor is just the instance count; the position-invariant numel rule (`ceil(numel(y) / inst_count)`, the busiest instance) gives the per-instance op count without re-deriving the reduced extent.
- **No per-call sampling state.** The block holds no fabricated mismatch, so the inherited `_sample_fabricate_mismatch` no-op is correct and `fabricate()` is a pass-through.
- **Energy and latency are two independent profiler emissions** (`_log_dynamic_energy` then `_log_latency`); the energy tensor is per-output-element while the latency is a single scalar scaled by the serial-op count. A consumer must not assume one event carries both.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants (`bit_width`, energy, latency) fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** A sum exceeding $[-2^{\,w-1}, 2^{\,w-1}-1]$ aliases with no error or warning; this is the intended hardware behaviour, not a bug, but a caller treating the block as saturating will get wrong results.
- **`inst_count` guarded against zero** in the divisor (`max(inst_count, 1)`); do not assume a positive instance count when reasoning about the serial-op math.

## Known limitations

- No dedicated unit test exists for the modular-wrap function or the PPA accounting; coverage is only indirect through higher-level macro tests.

---

- **Reference**: [accumulator](../../reference/digital/accumulator.md)
- **Implementation**: `neurox/digital/accumulator.py`
- **Tests**: TODO - no dedicated digital test module yet
- **Decisions**: N/A — no ADR governs this module.

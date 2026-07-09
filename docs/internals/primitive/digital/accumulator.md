# Accumulator

## Design decisions

- **Not part of a registry family.** The four digital blocks share no polymorphic dispatch surface, so they stay leaf circuits instantiated by concrete class name and are kept out of `RegistryMixin`. Adding family dispatch would impose a config-type discriminator with no caller that needs to select among them at runtime.

## Contracts & invariants

- **`operate(x, dim)` reduces exactly one axis**, so `numel(y)` already excludes the reduced extent and the serial-op divisor is the bare `inst_count`.
- **Stateless, single-call reduction.** One `operate` reduces the whole axis in a single batched reduce; the block carries no running total or register state across calls, despite the name.
- **Energy and latency are two independent profiler emissions** (`_log_dynamic_energy` then `_log_latency`): the energy tensor is per-output-element while the latency is a single scalar scaled by the serial-op count. One event does not carry both.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants (`bit_width`, energy, latency) fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; a caller treating the block as saturating gets wrong results.
- **`inst_count` guarded against zero** in the divisor (`max(inst_count, 1)`); do not assume a positive instance count when reasoning about the serial-op math.

## Known limitations

- No dedicated unit test exists for the modular-wrap function or the PPA accounting; coverage is only indirect through higher-level macro tests.

---

- **Reference**: [accumulator](../../../reference/primitive/digital/accumulator.md)
- **Implementation**: `neurox/digital/accumulator.py`
- **Tests**: TODO - no dedicated digital test module yet

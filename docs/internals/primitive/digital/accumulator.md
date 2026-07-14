# Accumulator

## Contracts & invariants

- **`operate(x, dim)` reduces exactly one axis**, so `numel(y)` already excludes the reduced extent and the serial-op divisor is the bare `inst_count`.
- **Stateless, single-call reduction.** One `operate` reduces the whole axis in a single batched reduce; the block carries no running total or register state across calls, despite the name.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants (`bit_width`, energy, latency) fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; a caller treating the block as saturating gets wrong results.

## Known limitations

- No dedicated unit test exists for the modular-wrap function or the PPA accounting; coverage is only indirect through higher-level macro tests.

---

- **Reference**: [accumulator](../../../reference/primitive/digital/accumulator.md)
- **Implementation**: `neurox/digital/accumulator.py`
- **Tests**: TODO - no dedicated digital test module yet

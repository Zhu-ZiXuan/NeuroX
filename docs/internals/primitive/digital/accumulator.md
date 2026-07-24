# Accumulator

## Contracts & invariants

- **`operate(x, dim)` reduces exactly one axis**, so `numel(y)` already excludes the reduced extent and the serial-op divisor is the bare `inst_count`.
- **Stateless, single-call reduction.** One `operate` reduces the whole axis in a single batched reduce; the block carries no running total or register state across calls, despite the name.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants (`bit_width`, energy, latency) fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; the operation is not saturating.

## Known limitations

- No standalone accumulator test module exists; the modular-wrap function and the per-output billing are covered directly by the billing-contrast tests in `tests/primitive/digital/test_serial_accumulator.py`.

---

- **Reference**: [accumulator](../../../reference/primitive/digital/accumulator.md)
- **Implementation**: `neurox/primitive/digital/accumulator.py`
- **Tests**: `tests/primitive/digital/test_serial_accumulator.py` (billing contrast)

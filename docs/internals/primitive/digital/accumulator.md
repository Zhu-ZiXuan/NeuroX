# Accumulator

## Contracts & invariants

- **`accumulate(x, dim)` reduces exactly one axis**, and the billed tensor is the pre-reduction operand, so the reduced extent stays inside the energy element count.
- **Stateless, single-call reduction.** One `accumulate` reduces the whole axis in a single batched reduce; the block carries no running total or register state across calls, despite the name.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants (`bit_width`, energy, latency) fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; the operation is not saturating.

## Known limitations

- No standalone accumulator test module exists; the modular-wrap function and the per-operand billing are covered directly by the tests in `tests/primitive/digital/test_serial_accumulator.py`, which exercise both accumulators against the shared law.

---

- **Reference**: [accumulator](../../../reference/primitive/digital/accumulator.md)
- **Implementation**: `neurox/primitive/digital/accumulator.py`
- **Tests**: `tests/primitive/digital/test_serial_accumulator.py` (shared billing law)

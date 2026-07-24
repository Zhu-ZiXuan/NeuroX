# Serial accumulator

## Contracts & invariants

- **`accumulate(x, dim)` reduces exactly one time-serial axis** with the same modular-wrap function as the accumulator; only the billing differs — one energy quantum per input element and a serial-round count of `ceil(numel(x) / inst_count)`, where the accumulator bills against the output-element count. The full per-input energy tensor is created only while a profiler is active.
- **Stateless, single-call reduction.** The serial-register semantics live only in the accounting; the reduce itself is one batched kernel with no cross-call state.
- **Config is `AccumulatorConfig`, reused unchanged.** The subclass adds no fields; the per-op terms are re-read as per-input-element quantities.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; the operation is not saturating.
- **Do not use for parallel adder trees.** A reduction realized as a parallel tree carries the accumulator's per-output billing; this block would over-bill it by the reduced-axis extent.

---

- **Reference**: [serial_accumulator](../../../reference/primitive/digital/serial_accumulator.md)
- **Implementation**: `neurox/primitive/digital/serial_accumulator.py`
- **Tests**: `tests/primitive/digital/test_serial_accumulator.py`

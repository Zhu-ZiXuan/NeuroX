# Serial accumulator

## Contracts & invariants

- **`accumulate(x, dim)` reduces exactly one time-serial axis**, inherited verbatim from the accumulator — same modular wrap, same per-operand billing off `x` before the reduce. The subclass overrides nothing; it records at the construction site that the reduced axis is realized as successive arrivals on one register, which is why the caller multiplies the per-op window by the arrival count in its own latency.
- **Stateless, single-call reduction.** The serial-register semantics live only in the caller's schedule; the reduce itself is one batched kernel with no cross-call state.
- **Config is `AccumulatorConfig`, reused unchanged.** The subclass adds no fields and re-reads no term.

## Performance & resources

- The reduction and the modular wrap are a single shape-clean kernel; per-op constants fold as compile-time constants under `@torch.compile`, so no graph break is introduced by the block.

## Gotchas

- **Wrap is silent.** An out-of-range sum aliases with no error or warning; the operation is not saturating.
- **The class carries no behavioural delta.** Energy, latency and function are the accumulator's; the name records the realization for the reader and nothing more, so swapping it for its base changes no reported number.

---

- **Reference**: [serial_accumulator](../../../reference/primitive/digital/serial_accumulator.md)
- **Implementation**: `neurox/primitive/digital/serial_accumulator.py`
- **Tests**: `tests/primitive/digital/test_serial_accumulator.py`

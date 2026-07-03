# Subtractor

## Summary

`Subtractor` computes an element-wise integer difference of two broadcastable tensors and emits its own PPA events. It is the sign twin of [adder](adder.md). Spec: [reference/digital/subtractor](../../reference/digital/subtractor.md).

## Design decisions

- **Behavioural, not gate-level.** Same rationale as the [adder](adder.md): the block models the function and a flat per-op cost, not a borrow-chain netlist, because the digital periphery is not the fidelity-critical path.
- **Kept a separate class, not a flag on the adder.** The subtract is a distinct operation with its own minuend/subtrahend argument semantics and its own config and PPA numbers; folding it into the adder via a sign flag would overload one call signature and conflate two cost models. The duplication is intentional and minimal.
- **No wrap, `bit_width` informational.** As with the adder, neither saturation nor modular wrap is applied; `bit_width` is metadata for PPA sizing only.

## Contracts & invariants

- **`operate(a, b)` computes `a - b` element-wise with broadcasting**; `a` is the minuend, `b` the subtrahend. The serial-op count uses the broadcast output numel divided by the instance count.
- **No per-call sampling state.** The `DigitalCircuit` base no-op is correct; `fabricate()` is a pass-through.
- **Energy and latency are two independent profiler emissions.**

## Performance & resources

- A single element-wise kernel; per-op constants fold under `@torch.compile` with no graph break.

## Gotchas

- **Operand order is load-bearing.** Unlike the adder, the subtract is not commutative; swapping `a` and `b` negates the result. The minuend is the first argument.
- **No range guard**, and **`inst_count` guarded against zero** in the divisor (`max(inst_count, 1)`).

## Known limitations

- No dedicated unit test for the function or the PPA accounting; coverage is only indirect.

---

- **Reference**: [subtractor](../../reference/digital/subtractor.md)
- **Implementation**: `neurox/digital/subtractor.py`
- **Tests**: TODO - no dedicated digital test module yet
- **Decisions**: N/A — no ADR governs this module.

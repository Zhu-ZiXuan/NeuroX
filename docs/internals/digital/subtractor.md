# Subtractor

## Design decisions

- **Behavioural, not gate-level.** The digital periphery is not the fidelity-critical path, so the block carries the flat per-op cost model specified in Reference rather than a gate-level borrow chain.
- **A separate class, not a flag on the adder.** The subtract has its own minuend/subtrahend argument order and its own config and PPA numbers; folding it into the adder via a sign flag would overload one call signature and conflate two cost models. The duplication is intentional and minimal.

## Contracts & invariants

- **`operate(a, b)`** takes `a` and a broadcast-compatible `b` and returns their difference at the broadcast shape.
- **Energy and latency are two independent profiler emissions.**

## Performance & resources

- A single element-wise kernel; the per-op constants fold under `@torch.compile` with no graph break.

## Gotchas

- **Operand order is load-bearing.** The first argument is the minuend; swapping the operands negates the result.
- **No range guard**: a difference exceeding the nominal bit width is not detected. **`inst_count` is guarded against zero** in the serial-op divisor (`max(inst_count, 1)`).

## Known limitations

- No dedicated unit test for the function or the PPA accounting; coverage is only indirect.

---

- **Reference**: [subtractor](../../reference/digital/subtractor.md)
- **Implementation**: `neurox/digital/subtractor.py`
- **Tests**: TODO — no dedicated digital test module yet

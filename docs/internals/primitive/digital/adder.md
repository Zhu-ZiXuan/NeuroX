# Adder

## Contracts & invariants

- **`add(a, b)` is element-wise with PyTorch broadcasting**; the output shape is the broadcast of the two operands.

## Performance & resources

- A single element-wise kernel; per-op constants fold under `@torch.compile` with no graph break.

## Gotchas

- **No runtime range check.** `add` enforces no bound on the operand sum; a value exceeding the nominal `bit_width` passes through undetected.

## Known limitations

- No dedicated unit test for the function or the PPA accounting; coverage is only indirect.

---

- **Reference**: [adder](../../../reference/primitive/digital/adder.md)
- **Implementation**: `neurox/primitive/digital/adder.py`
- **Tests**: TODO - no dedicated digital test module yet

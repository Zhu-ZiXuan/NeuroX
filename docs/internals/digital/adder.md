# Adder

## Design decisions

- **Not part of a registry family.** Shares no polymorphic dispatch with the other digital blocks, so it stays a leaf circuit and is kept out of `RegistryMixin`.

## Contracts & invariants

- **`operate(a, b)` is element-wise with PyTorch broadcasting**; the output shape is the broadcast of the two operands.
- **Energy and latency are two independent profiler emissions**; the energy tensor is per-output-element, the latency a single scaled scalar.

## Performance & resources

- A single element-wise kernel; per-op constants fold under `@torch.compile` with no graph break.

## Gotchas

- **No runtime range check.** `operate` enforces no bound on the operand sum; a value exceeding the nominal `bit_width` passes through undetected.
- **`inst_count` guarded against zero** in the divisor (`max(inst_count, 1)`).

## Known limitations

- No dedicated unit test for the function or the PPA accounting; coverage is only indirect.

---

- **Reference**: [adder](../../reference/digital/adder.md)
- **Implementation**: `neurox/digital/adder.py`
- **Tests**: TODO - no dedicated digital test module yet

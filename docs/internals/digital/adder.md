# Adder

## Summary

`Adder` computes an element-wise integer sum of two broadcastable tensors and emits its own PPA events. Spec: [reference/digital/adder](../../reference/digital/adder.md).

## Design decisions

- **Behavioural, not gate-level.** The block models the function and a flat per-op cost, not a netlist; there is no carry-chain or per-bit timing model, because the digital periphery is not the fidelity-critical path and a netlist would add accuracy nothing downstream consumes.
- **No wrap, `bit_width` informational.** Unlike the accumulator and shift-adder, the adder applies neither saturation nor modular wrap: `bit_width` is metadata for PPA sizing only. The rationale is that an element-wise add of values the consumer has already range-checked needs no register fold; imposing one here would double-wrap when the result later flows into an accumulator.
- **Not part of a registry family.** Shares no polymorphic dispatch with the other digital blocks, so it stays a leaf circuit and is kept out of `RegistryMixin`.

## Contracts & invariants

- **`operate(a, b)` is element-wise with PyTorch broadcasting**; the output shape is the broadcast of the two operands. The serial-op count uses the broadcast output numel divided by the instance count (position-invariant numel rule).
- **No per-call sampling state.** Inherited `_sample_fabricate_mismatch` no-op is correct; `fabricate()` is a pass-through.
- **Energy and latency are two independent profiler emissions**; the energy tensor is per-output-element, the latency a single scaled scalar.

## Performance & resources

- A single element-wise kernel; per-op constants fold under `@torch.compile` with no graph break.

## Gotchas

- **No range guard.** Because `bit_width` is informational, an operand sum that exceeds the nominal width is not detected; the consumer owns range correctness.
- **`inst_count` guarded against zero** in the divisor (`max(inst_count, 1)`).

## Known limitations

- No dedicated unit test for the function or the PPA accounting; coverage is only indirect.

---

- **Reference**: [adder](../../reference/digital/adder.md)
- **Implementation**: `neurox/digital/adder.py`
- **Tests**: TODO - no dedicated digital test module yet
- **Decisions**: N/A — no ADR governs this module.

# Shift-adder

`ShiftAdder` folds a digit axis with radix-positional weights into a fixed-width signed register, optionally adds a running partial sum, and emits its own PPA events.

## Design decisions

- **Behavioural, not gate-level.** The block models the radix-fold function and a flat per-op cost, not a shift-and-add netlist, because the digital periphery is not the fidelity-critical path.
- **Radix and partial sum are call arguments, not config.** The radix `scale`, the digit axis `dim`, and the optional partial sum `init_val` are passed per call rather than fixed in the config. Only the cost terms and the register width are construction-time constants.
- **Partial sum added after the wrap, not before.** `init_val` is summed onto the wrapped radix-fold result, so a running accumulator can carry a total past the per-call register range. Folding it in before the wrap would clip the running total to one call's register and break chaining; this ordering is correctness-relevant.

## Contracts & invariants

- **`shift_add(x, scale, dim, init_val)` reduces exactly the `dim` axis**; the radix weights are built on `x`'s device and dtype. The reduced axis is gone from the output, so the serial-op divisor is the instance count.
- **`init_val` must broadcast to the reduced output shape** (post-reduction, digit axis removed), not to the input shape.
- **No per-call sampling state.** The shift-adder holds no fabricated mismatch, so the base fabricate no-op ([base](base.md)) applies unchanged.

## Performance & resources

- The weight-vector construction, the weighted sum, the modular wrap, and the optional partial-sum add are a single shape-clean kernel. The per-op constants fold under `@torch.compile`, but `scale` and `x.size(dim)` are runtime values, so the radix-weight construction may specialize or guard on the digit count rather than fold unconditionally.

## Gotchas

- **Post-wrap partial-sum semantics.** Because `init_val` is added after the wrap, the final output is not confined to the register range when a partial sum is supplied.
- **The radix-fold sum wraps silently** before the partial-sum add; an overflow inside one call aliases with no error.

## Known limitations

- The flat per-op latency does not grow with the digit count $D$; if a hardware mapping serializes the shift-add across digits, the model under-counts latency. No dedicated unit test for the function or the PPA accounting exists; coverage is only indirect.

---

- **Reference**: [shift_adder](../../../reference/primitive/digital/shift_adder.md)
- **Implementation**: `neurox/primitive/digital/shift_adder.py`
- **Tests**: TODO - no dedicated digital test module yet

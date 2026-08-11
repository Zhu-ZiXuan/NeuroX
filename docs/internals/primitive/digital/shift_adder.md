# Shift-adder

`ShiftAdder` folds a digit axis with radix-positional weights into a fixed-width signed register, optionally adds a running partial sum, and emits its own PPA events.

## Design decisions

- **Behavioural, not gate-level.** The block models the radix-fold function and a flat per-op cost, not a shift-and-add netlist, because the digital periphery is not the fidelity-critical path.
- **Positional geometry is bound at construction.** The radix `scale` and `digit_count` are explicit init arguments because one hardware instance implements one fixed recombination geometry. The digit axis `dim` and optional partial sum `init_val` remain call arguments.
- **Partial sum added after the wrap, not before.** `init_val` is summed onto the wrapped radix-fold result, so a running accumulator can carry a total past the per-call register range. Folding it in before the wrap would clip the running total to one call's register and break chaining; this ordering is correctness-relevant.

## Contracts & invariants

- **`shift_add(x, dim, init_val)` reduces exactly the `dim` axis.** The fixed radix weights are an int64 functional buffer, so they follow module device migration and are not reconstructed in the execution path. The reduced axis is gone from the output but stays in the bill, which is read off `x` before the reduce.
- **`init_val` must broadcast to the reduced output shape** (post-reduction, digit axis removed), not to the input shape.
- **No per-call sampling state.** The shift-adder holds no fabricated mismatch, so the base fabricate no-op ([base](base.md)) applies unchanged.

## Performance & resources

- The weighted sum, modular wrap, and optional partial-sum add are a shape-clean kernel. Construction materializes the fixed positional-weight vector once; execution introduces no Python-to-Tensor conversion for it.

## Gotchas

- **Post-wrap partial-sum semantics.** Because `init_val` is added after the wrap, the final output is not confined to the register range when a partial sum is supplied.
- **The radix-fold sum wraps silently** before the partial-sum add; an overflow inside one call aliases with no error.

## Known limitations

- The digit axis is modelled as space: the $D$ legs are weighted and summed in one pass, so the per-op window is flat in $D$ while the energy scales with it. A mapping that serializes the legs is the scheduling caller's time axis, and that caller counts it.

---

- **Reference**: [shift_adder](../../../reference/primitive/digital/shift_adder.md)
- **Implementation**: `neurox/primitive/digital/shift_adder.py`
- **Tests**: `tests/primitive/digital/test_shift_adder.py`

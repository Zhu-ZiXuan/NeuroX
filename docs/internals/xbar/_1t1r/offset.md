# Offset1T1RXbar — Implementation

## Summary

`Offset1T1RXbar` (`_1t1r/offset.py`) composes the shared [circuit_core](circuit_core.md) with a [readout](../readout/README.md) chain into the offset-coded operating xbar. Spec: [reference/xbar/_1t1r/offset](../../../reference/xbar/_1t1r/offset.md).

## Design decisions

- **`vec_mat_mul` is `@torch.compiler.disable`d.** The macro-level `@torch.compile` would otherwise trace into the readout's SAR-ADC bit-loop and hit graph breaks (>10 min compile). Disabling isolates the inner numeric block; the compiled boundary stays at the macro entry.
- **Chunk knobs live on `CircuitCore1T1RPolicy`, not on the chip config.** They depend on the host GPU budget, not chip physics, so one chip TOML is reused across hosts with per-host chunk sizes; the offset policy forwards them down to the core.

## Contracts & invariants

- **`program(w)` does the logical-to-physical scatter then delegates.** It applies the data-major / digit-minor column scatter plus reference-column insertion and the offset shift, then calls `circuit_core.program(...)`; the core stays encoding-agnostic.

## Performance & resources

N/A at this level — the chunked solve and its memory model are in [circuit_core](circuit_core.md) and [solver](solver.md).

## Gotchas

- N/A.

## Known limitations

- Block-level `@torch.compile` on the readout is not enabled: inductor scheduling of the SAR-ADC bit-loop produces >10 min compile times. Re-enabling requires rewriting the SAR ADC into a graph-friendly form first.

---

- **Reference**: [offset](../../../reference/xbar/_1t1r/offset.md)
- **Implementation**: `neurox/xbar/_1t1r/offset.py`
- **Tests**: `tests/test_xbar_physics.py`
- **Decisions**: N/A — no ADR governs this module.

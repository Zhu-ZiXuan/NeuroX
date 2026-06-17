# Offset1T1RXbar — Implementation

## Summary

`Offset1T1RXbar` (`_1t1r/offset.py`) composes the shared [circuit_core](circuit_core.md) with a [readout](../readout/README.md) chain into the offset-coded operating xbar. Spec: [reference/xbar/_1t1r/offset](../../../reference/xbar/_1t1r/offset.md).

## Design decisions

- **`vec_mat_mul` runs eager; the one self-compiled region is a level down.** The library does not self-compile the macro forward, so this method's index / readout math runs eager — and stays compile-friendly so a caller may `torch.compile` the model. The data-dependent chunk loop lives in [circuit_core](circuit_core.md)'s `cim_read` (`@torch.compiler.disable`), and the DC-solve bottleneck is the one self-compiled fixed-shape leaf (`solve_dc`). See [compile/scheme-a-regional](../../compile/scheme-a-regional.md).
- **The chunk knob lives on `CircuitCore1T1RPolicy`, not on the chip config.** `solve_chunk_size` depends on the host GPU budget, not chip physics, so one chip TOML is reused across hosts with a per-host chunk size; the offset policy forwards it down to the core.

## Contracts & invariants

- **`program(w)` does the logical-to-physical scatter then delegates.** It applies the data-major / digit-minor column scatter plus reference-column insertion and the offset shift, then calls `circuit_core.program(...)`; the core stays encoding-agnostic.

## Performance & resources

N/A at this level — the chunked solve and its memory model are in [circuit_core](circuit_core.md) and [solver](solver.md).

## Gotchas

- N/A.

## Known limitations

- N/A.

---

- **Reference**: [offset](../../../reference/xbar/_1t1r/offset.md)
- **Implementation**: `neurox/xbar/_1t1r/offset.py`
- **Tests**: `tests/test_xbar_physics.py`
- **Decisions**: N/A — no ADR governs this module.

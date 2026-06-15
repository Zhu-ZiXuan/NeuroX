# Readout — Implementation

## Summary

The `ReadOut` container (`readout/base.py`, `readout/offset_switchcap_mux_adc.py`) orchestrates the voltage-domain chain; the leaf blocks (switch-cap, mux, ADC) own the electrical math. Spec: [reference/xbar/readout](../../../reference/xbar/readout/README.md).

## Design decisions

- **The container does shape movement and energy aggregation only** — no signal-value arithmetic outside the leaf switch-cap kernels. Keeping all electrical math in the leaves means the container never silently re-implements a weighted sum or a baseline.
- **Exposed as `readout(...)`, not `forward()`.** The chain is non-trainable and never participates in autograd, so the explicit call skips `nn.Module` hook plumbing and keeps the call site clear.

## Contracts & invariants

- Children are constructed with a derived `inst_shape` and `fabricate()` auto-cascades into them; the container registers no electrical state of its own.

## Performance & resources

N/A — the readout is not on the memory- or compile-critical path (see [_1t1r/offset](../_1t1r/offset.md) for the SAR-ADC compile limitation).

## Gotchas

- N/A.

## Known limitations

- N/A.

---

- **Reference**: [readout](../../../reference/xbar/readout/README.md)
- **Implementation**: `neurox/xbar/readout/base.py`, `neurox/xbar/readout/offset_switchcap_mux_adc.py`
- **Tests**: `tests/test_xbar_physics.py`
- **Decisions**: N/A — no ADR governs this module.

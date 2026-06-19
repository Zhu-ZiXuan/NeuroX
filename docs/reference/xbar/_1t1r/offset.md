# Offset 1T1R Xbar

## Summary

`Offset1T1RXbar` is an operating xbar: it composes the shared [circuit_core](circuit_core.md) with a [readout](../readout/README.md) chain and an offset coding scheme to turn the physical array into a signed VMM. A future differential 1T1R xbar is a sibling that reuses the same `circuit_core` with a different coding and readout. The abstract VMM contract, value domain, and output rescale are in [base](../base.md).

## Offset coding

A value maps to physical columns in slice-major / digit-minor order, with a reference column inserted per group. The precondition is that the reference column encodes the coding zero point: it is programmed so that its column result equals the contribution the data column would produce at the coding zero point, so the per-group subtraction cancels the structural coding offset exactly. The read is differential (data minus reference), so this common-mode offset is removed and the signed result carries the sign of the dot product directly. Here common-mode offset means the structural offset-coding term shared by the data and reference legs — the deterministic shift introduced by the offset coding, not a noise offset. The value→physical scatter is applied at programming time; the differential subtraction at readout (see the [readout chain](../readout/README.md)). The code-to-$M_{\mathrm{ideal}}$ rescale convention is the abstract-layer contract in [base](../base.md#output-rescale).

## Symbols

As in [base](../base.md#symbols) ($D$, $r$, $s$, $b$) and [circuit_core](circuit_core.md#symbols); offset coding adds no new physical symbol.

## Assumptions, scope & validity

Stated assumption: the reference column encodes the coding zero point, so the differential data-minus-reference read cancels the structural offset-coding common-mode offset (not a noise offset); only the differential signal is digitized.

TODO (domain author): the validity boundary of the offset-cancellation assumption (data-vs-reference leg mismatch, ADC input-range limits).

## Validation

TODO: link [validation/xbar](../../../validation/README.md) — physical-vs-ideal code agreement under a noise-off policy.

## References

TODO: cite the offset / reference-column coding scheme.

---

- **Internals**: [offset internals](../../../internals/xbar/_1t1r/offset.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../api/README.md) (`[xbar]`)
- **Decisions**: N/A — no ADR governs this module.

# Offset 1T1R Xbar

## Summary

`Offset1T1RXbar` is an operating xbar: it composes the shared [circuit_core](circuit_core.md) with a [readout](../readout/README.md) chain and an offset coding scheme to turn the physical array into a signed VMM. A future differential 1T1R xbar is a sibling that reuses the same `circuit_core` with a different coding and readout. The abstract VMM contract, value domain, and output rescale are in [base](../base.md).

## Offset coding

A logical column maps to physical columns in data-major / digit-minor order, with a reference column inserted per group. The read is differential (data minus reference), so the reference cancels a common-mode offset and the signed result carries the sign of the dot product directly. The logical→physical scatter is applied at programming time; the differential subtraction at readout (see the [readout chain](../readout/README.md)). The code-to-$M_{\text{ideal}}$ rescale convention is the abstract-layer contract in [base](../base.md#output-rescale).

## Symbols

As in [base](../base.md#symbols) ($D$, $r$, $s$, $b$) and [circuit_core](circuit_core.md#symbols); offset coding adds no new physical symbol.

## Assumptions, scope & validity

Stated assumption: the differential data-minus-reference read cancels a common-mode offset; only the differential signal is digitized.

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

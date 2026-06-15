# Readout Chain

## Summary

The readout family is the voltage-domain chain from an array's per-column boundary output to signed ADC codes. An operating xbar (e.g. [offset](../_1t1r/offset.md)) composes a readout with its core; the abstract output-rescale contract is in [base](../base.md#output-rescale). The current implementation is an offset switch-cap / mux / differential ADC chain.

## Physical model

The chain operates on a grouped lattice keyed by reference-group structure (`group_num` reference groups, each with `data_num` data and $D$ digits) and is built from leaf analog blocks; all electrical signal transformation happens inside the leaves, while the chain only moves and aggregates:

- **data switch-cap** — accumulates the per-digit voltages with positional capacitor weights (`digit_weights`, one cap per digit);
- **reference switch-cap** — samples a single unit-cap baseline per group;
- **analog mux** — transports the grouped signals;
- **differential ADC** — converts the data leg against the reference leg.

The leaf models are specified in [reference/analog](../../analog/README.md) (switch-cap, mux, ADC).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $D$ | digits per data | — | `digit_num` (= the tile's `w_digit_count`) |

Structure counts are referred to by code field (`group_num`, `data_num`, `digit_weights`); the value-domain symbols ($M_{\text{ideal}}$, $s$, $b$) are in [base](../base.md#symbols).

## Noise & non-idealities

ADC quantization is intrinsic; further non-idealities enter through the leaf blocks (ADC offset/comparator, switch-cap, mux), each gated by its policy — see [reference/analog](../../analog/README.md). TODO: enumerate the offset / quantization / charge-injection sources that survive the differential cancellation and how they perturb the code.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `data_num` | data per reference group | — | Design |
| `digit_weights` | per-digit positional weights | — | Design |
| `group_num` | reference groups | — | Design |
| data / ref switch-cap, mux, ADC configs | leaf-block configs | — | Design |
| `adc_calibration` ($s$ table) | ADC (mode, $b$) to $s$ lookup | — | Calibrated (physical data) |

Provenance terms: [parameter_provenance](../../parameter_provenance.md). Schema: [config reference](../../../api/README.md); rescale calibration: [calibration guide](../../../guides/calibration/README.md).

## Assumptions, scope & validity

Stated assumption: all signal-value transformation is confined to the leaf blocks; the chain performs no weighting or baseline arithmetic outside the switch-cap kernels.

TODO (domain author): neglected charge-injection / settling effects and the ADC input-range limits.

## Validation

TODO: link [validation/xbar](../../../validation/README.md) — ADC-boundary calibration evidence.

## References

TODO: cite the differential switch-cap readout.

---

- **Internals**: [readout internals](../../../internals/xbar/readout/README.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../api/README.md) (`[xbar.readout_config]`, `adc_calibration`)
- **Decisions**: N/A — no ADR governs this module.

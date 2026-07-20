# isub_iadc_1t1r documentation

A single-tile 1T1R CIM macro with a current-subtractor / current-ADC readout: ternary weights {-1, 0, +1} on P/N physical column pairs, 1-bit inputs, and a multi-mode signed-magnitude output code per WL plane (sign bit + `n_bits` ADC magnitude; the width is set by `adc_config.n_bits` in the params) — one conversion drives at most `active_row_num` rows, the consuming engine serializes a full-row read into `row_num / active_row_num` pre-masked WL planes, and the tile emits per-plane codes without in-macro accumulation. The tile composes only shared kernel primitives (`neurox/primitive/`) — the pure array is the kernel `XbarArray1t1r`, with the boundary drivers macro-owned as its peers; there are no scheme-local circuit classes and no paper-acronym classes — the paper's block names appear only as provenance in prose.

## Documents

| Document | Scope |
| --- | --- |
| [reference/macro.md](reference/macro.md) | Scientific spec of the tile: per-plane read sequence, boundary drive, transfer math, value domain, sharing geometry, energy ownership |
| [internals/macro.md](internals/macro.md) | Tile implementation: tensor shapes, array-solve stage, reshape-broadcast sharing, accounting ownership |

## Relation to the paper

The readout chain follows Xue et al., IEEE JSSC 2020 (the embedded 1-Mb ReRAM macro).

- **Modeled 1:1** — the current-mode read chain topology: current-aware BL clamp (the paper's CABLC slot), front-end down-scale mirror (DSWCT slot), back-end normalization mirror (combiner slot), P-N current subtractor (PN-ISUB slot), SAR current ADC with mid-point thresholds from a shared static reference (TMCSA + Reference slots); column-MUX time-sharing of the readout circuits; binary-mode conversion timing anchor; partial-row activation with a small per-conversion code domain.
- **Intentionally absent** — input bit-serial sub-cycles (one binary WL plane per conversion), sample-and-hold capacitors, and weight MSB/LSB digit machinery (a single ternary digit per weight).

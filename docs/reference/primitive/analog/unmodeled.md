# Unmodeled block

## Physical model

A real circuit block whose internal transfer is out of modelling scope, represented only by static PPA. It reserves silicon area and standing bias power while deliberately omitting the data transfer and per-operation dynamic energy.

## Governing equations

N/A — the block has no functional transfer. Its only reported quantities are static:

$$A = a_{\mathrm{inst}} \cdot N, \qquad P_{\mathrm{leak}} = p_{\mathrm{inst}} \cdot N,$$

with $a_{\mathrm{inst}}$ / $p_{\mathrm{inst}}$ the per-instance area / leakage and $N$ the instance count.

## Numerical method

N/A — no computation.

## Noise & non-idealities

None modelled. The block is functionally unmodeled and has no policy sources.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `area_per_inst__um2` ($a_{\mathrm{inst}}$) | silicon area per fabricated instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` ($p_{\mathrm{inst}}$) | static leakage per instance (carries the block's whole standing bias power) | uW | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $a_{\mathrm{inst}}$ | per-instance area | um^2 | `area_per_inst__um2` |
| $p_{\mathrm{inst}}$ | per-instance leakage | uW | `leakage_per_inst__uW` |
| $N$ | instance count | — | `inst_count` |

## Assumptions, scope & validity

The block's leakage is a constant static draw independent of the signal. It has no functional transfer or dynamic-energy model.

## Validation

TODO - link validation evidence once written.

## References

N/A.

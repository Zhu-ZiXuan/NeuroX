# Unmodeled block

## Physical model

A real circuit block whose internal transfer is out of modelling scope, represented by lumped area, standing power, and per-operation dynamic energy. It reserves a PPA seat without asserting an internal circuit decomposition or data-transfer law.

## Governing equations

The block has no functional transfer. Its lumped quantities are

$$A = a_{\mathrm{inst}} \cdot N, \qquad P_{\mathrm{leak}} = p_{\mathrm{inst}} \cdot N, \qquad E_{\mathrm{dyn}} = e_{\mathrm{op}} \cdot N_{\mathrm{op}},$$

with $a_{\mathrm{inst}}$ and $p_{\mathrm{inst}}$ the per-instance area and leakage, $N$ the instance count, $e_{\mathrm{op}}$ the energy per modeled operation, and $N_{\mathrm{op}}$ the number of operations billed.

## Numerical method

N/A — no internal transfer is evaluated.

## Noise & non-idealities

None modelled. The block is functionally unmodeled and has no policy sources.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `area_per_inst__um2` ($a_{\mathrm{inst}}$) | silicon area per fabricated instance | um^2 | $\geq 0$ | Design |
| `leakage_per_inst__uW` ($p_{\mathrm{inst}}$) | static leakage per instance (carries the block's whole standing bias power) | uW | $\geq 0$ | Design |
| `energy_per_op__fJ` ($e_{\mathrm{op}}$) | lumped dynamic energy per modeled operation | fJ | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $a_{\mathrm{inst}}$ | per-instance area | um^2 | `area_per_inst__um2` |
| $p_{\mathrm{inst}}$ | per-instance leakage | uW | `leakage_per_inst__uW` |
| $e_{\mathrm{op}}$ | dynamic energy per modeled operation | fJ | `energy_per_op__fJ` |
| $N$ | instance count | — | `inst_count` |
| $N_{\mathrm{op}}$ | operation count | — | — |

## Assumptions, scope & validity

The leakage is a constant static draw independent of the signal. The dynamic term is a flat event cost independent of data values; no internal nodes or transfer behavior are represented.

## Validation

TODO - link validation evidence once written.

## References

N/A.

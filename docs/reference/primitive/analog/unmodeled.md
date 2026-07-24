# Unmodeled block

## Physical model

A real circuit block whose internal transfer is out of modelling scope, kept on the ledger only as static PPA. It reserves the block's silicon area and its whole standing bias power (a static, always-on draw, not derived from any signal) so the block's cost is not silently dropped, while its functional behaviour — the data-in/data-out transfer — is deliberately not modelled. Typical uses: control logic, a fixed bias network, or any peripheral whose transfer does not affect the modelled signal path but whose power must still be counted. Any per-op dynamic energy the block draws is not billed here; a composing macro that knows the block's activity bills it through a profiler channel keyed to the block's role.

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

The block's leakage is a constant static draw, independent of the signal. It has no functional method and no state, so it never appears in the modelled signal path. It is a profile target: its static PPA is visible to the profiler's static walk and scales by instance count. When the block does draw data-dependent dynamic energy, that energy is the composing macro's responsibility to bill under a named channel — this block never emits a dynamic event itself.

## Validation

TODO - link validation evidence once written.

## References

N/A.

---

- **Internals**: [unmodeled internals](../../../internals/primitive/analog/unmodeled.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `UnmodeledBlockConfig`, `UnmodeledBlockPolicy` (see `api`)

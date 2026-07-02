# Monotonic SAR ADC

## Summary / role

Note: this document is incomplete and not yet finished. Its symbols, units and equations are provisional and are not reconciled with the rest of the ADC family yet; do not treat it as a settled spec.

`SarAdcMono` is the monotonic (Set-and-Down) differential SAR ADC, a planned member of the [ADC family](README.md). Its fabrication state (per-leg cap arrays and comparator offset) is specified and wired, but the differential conversion kernel is not yet realised. For current SAR work use [mcs_sar](mcs_sar.md). It will honour the family signed-code and floor contract in [base](base.md).

## Physical model

A differential SAR with monotonic Set-and-Down switching: each cycle the larger top plate drops by $V_{\mathrm{ref}}\,C_k/C_{\mathrm{total}}$ while the smaller side holds. Caps only ever discharge to ground during the SAR loop - once switched, a cap never re-charges to $V_{\mathrm{ref}}$. The modelled non-idealities mirror the MCS variant: static per-leg cap mismatch, static comparator offset, per-cycle comparator thermal noise, and kT/C sampling noise. The single-ended Set-and-Down variant is not modelled; only the differential one is planned.

## Governing equations

Each cycle steps the larger top plate down by the monotonic Set-and-Down step

$$\Delta V_{\mathrm{top},k} = -\,V_{\mathrm{ref}}\,\frac{C_k}{C_{\mathrm{total}}},$$

with $C_k$ the (mismatched, if the policy is on) capacitance switched on cycle $k$. The bit decisions and the unsigned-to-signed shift will follow the family contract in [base](base.md#signed-code-output-convention).

TODO (domain author): the full per-cycle decision and code-accumulation equations once the differential conversion kernel is realised.

## Numerical method

The planned conversion is a successive-approximation loop of closed-form Set-and-Down steps and sign decisions, structurally as in [mcs_sar](mcs_sar.md#numerical-method).

TODO (domain author): specify once the kernel exists; the conversion entry currently raises `NotImplementedError`.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| cap mismatch | per-cap area/oxide variation | static per-cap Pelgrom Gaussian on $C_k$ (independent legs), at fabricate | `cap_mismatch_sigma_relative` | `cap_mismatch` |
| comparator offset | static comparator input offset | static Gaussian threshold offset, at fabricate | `comparator_offset_sigma__V` | `comparator_offset` |
| comparator thermal noise | per-decision thermal noise | additive Gaussian per SAR cycle | `comparator_thermal_noise_sigma__V` | `comparator_thermal_noise` |
| sampling thermal noise | kT/C noise on held plates | additive Gaussian at sample | (derived from $T$, $C$) | `sampling_thermal_noise` |

The fabricated state consuming these (`c_p__fF` / `c_n__fF` per-leg cap arrays and the comparator offset) is wired; the per-cycle noise consumption awaits the conversion kernel.

## Parameters

Same parameter shape as [mcs_sar](mcs_sar.md#parameters): `max_bits`, `c_unit__fF`, `clk_period__ns`, the cap- and comparator-mismatch sigmas, energy overhead, and static PPA. As with the MCS variant the reference ladder is not a config field — the future kernel will read the per-call injected `v_refs__V` tensor the owning xbar sources and select a tap by $\mathrm{mode}$. Like the MCS variant there is no per-op-latency parameter; the future kernel will derive $(b+1)\cdot$ `clk_period__ns`. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumptions:

- Monotonic switching: caps only discharge to ground during the loop; the larger plate steps down while the smaller holds.
- Only the differential variant is modelled; the single-ended Set-and-Down variant is not.

TODO (domain author): the validity boundary of the Set-and-Down step model once the kernel is realised.

## Validation

TODO - link validation evidence once the conversion kernel is realised.

## References

TODO: cite the monotonic (Set-and-Down) SAR switching scheme.

## Symbols

As in [mcs_sar](mcs_sar.md#symbols); the monotonic step replaces the merged-capacitor step. Per-leg cap arrays are the code fields `c_p__fF` / `c_n__fF`.

---

- **Internals**: [sar_mono internals](../../../internals/analog/adc/sar_mono.md)
- **Validation**: TODO - validation evidence not yet written (kernel not realised)
- **Configuration**: `SarAdcMonoConfig`, `SarAdcMonoPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.

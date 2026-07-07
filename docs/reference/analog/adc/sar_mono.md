# Monotonic SAR ADC

## Summary / role

Note: this specification is provisional — its equations and validation are not yet settled; do not treat it as a final spec.

`SarAdcMono` is the monotonic (Set-and-Down) differential SAR ADC, a member of the [ADC family](family.md). It honours the family signed-code and floor contract in [base](family.md).

## Physical model

A differential SAR with monotonic Set-and-Down switching: each cycle the larger top plate drops by $V_{\mathrm{ref}}\,C_k/C_{\mathrm{total}}$ while the smaller side holds. Caps only ever discharge to ground during the SAR loop - once switched, a cap never re-charges to $V_{\mathrm{ref}}$. The modelled non-idealities mirror the MCS variant: static per-leg cap mismatch, static comparator offset, per-cycle comparator thermal noise, and kT/C sampling noise.

## Governing equations

Each cycle steps the larger top plate down by the monotonic Set-and-Down step

$$\Delta V_{\mathrm{top},k} = -\,V_{\mathrm{ref}}\,\frac{C_k}{C_{\mathrm{total}}},$$

with $C_k$ the capacitance switched on cycle $k$, carrying the static per-leg Pelgrom cap mismatch. The bit decisions and the unsigned-to-signed shift follow the family contract in [base](family.md#signed-code-range).

TODO (domain author): the full per-cycle decision and code-accumulation equations.

## Numerical method

The conversion is a successive-approximation loop of closed-form Set-and-Down steps and sign decisions, structurally as in mcs_sar.

TODO (domain author): specify the SAR-loop mathematics.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| cap mismatch | per-cap area/oxide variation | static per-cap Pelgrom Gaussian on $C_k$ (independent legs), at fabricate | `cap_mismatch_sigma_relative` |
| comparator offset | static comparator input offset | static Gaussian threshold offset, at fabricate | `comparator_offset_sigma__V` |
| comparator thermal noise | per-decision thermal noise | additive Gaussian per SAR cycle | `comparator_thermal_noise_sigma__V` |
| sampling thermal noise | kT/C noise on held plates | additive Gaussian at sample | (derived from $T$, $C$) |

## Parameters

The parameters are those of mcs_sar: `max_bits`, `c_unit__fF`, `clk_period__ns`, the cap- and comparator-mismatch sigmas, energy overhead, and static PPA. The reference ladder is not a config field: the taps are injected per call (shape `(*inst, num_refs)`) and $\mathrm{mode}$ selects one. Per-op latency is likewise not a parameter: it is derived as $(b+1)\cdot$ `clk_period__ns` from the runtime operating point. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

As in mcs_sar; the monotonic step replaces the merged-capacitor step. Per-leg cap arrays are the code fields `c_p__fF` / `c_n__fF`.

## Assumptions, scope & validity

Stated assumptions:

- Monotonic switching: caps only discharge to ground during the loop; the larger plate steps down while the smaller holds.
- Only the differential variant is modelled; the single-ended Set-and-Down variant is not.

TODO (domain author): the validity boundary of the Set-and-Down step model.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the monotonic (Set-and-Down) SAR switching scheme.

---

- **Internals**: [sar_mono internals](../../../internals/analog/adc/sar_mono.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `SarAdcMonoConfig`, `SarAdcMonoPolicy` (see `api`)

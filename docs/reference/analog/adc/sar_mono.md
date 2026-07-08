# Monotonic SAR ADC

The monotonic (Set-and-Down) differential SAR ADC digitizes a differential input into a signed integer code, a member of the [ADC family](family.md) honouring the family signed-code and floor contract in [base](family.md).

## Physical model

A differential SAR with monotonic Set-and-Down switching: each cycle the larger top plate drops by $V_{\mathrm{ref}}\,C_k/C_{\mathrm{total}}$ while the smaller side holds. Caps only ever discharge to ground during the SAR loop - once switched, a cap never re-charges to $V_{\mathrm{ref}}$. The modelled non-idealities are static per-leg cap mismatch, static comparator offset, per-cycle comparator thermal noise, and kT/C sampling noise.

## Governing equations

Each cycle steps the larger top plate down by the monotonic Set-and-Down step

$$\Delta V_{\mathrm{top},k} = -\,V_{\mathrm{ref}}\,\frac{C_k}{C_{\mathrm{total}}},$$

with $C_k$ the capacitance switched on cycle $k$, carrying the static per-leg Pelgrom cap mismatch. The bit decisions and the unsigned-to-signed shift follow the family contract in [base](family.md#signed-code-range).

TODO (domain author): the full per-cycle decision and code-accumulation equations.

## Numerical method

The conversion is a successive-approximation loop of closed-form Set-and-Down steps and sign decisions.

TODO (domain author): specify the SAR-loop mathematics.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| cap mismatch | per-cap area/oxide variation | static per-cap Pelgrom Gaussian on $C_k$ (independent legs), at fabricate | `cap_mismatch_sigma_relative` |
| comparator offset | static comparator input offset | static Gaussian threshold offset, at fabricate | `comparator_offset_sigma__V` |
| comparator thermal noise | per-decision thermal noise | additive Gaussian per SAR cycle | `comparator_thermal_noise_sigma__V` |
| sampling thermal noise | kT/C noise on held plates | additive Gaussian at sample | (derived from $T$, $C$) |

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `max_bits` | physical CDAC depth $b_{\max}$ | — | integer, $\geq 2$ | Design |
| `clk_period__ns` | SAR comparator clock period | ns | $> 0$ | Design |
| `c_unit__fF` | CDAC unit capacitance | fF | $> 0$ | Design |
| `cap_mismatch_sigma_relative` | per-unit-cap relative Pelgrom sigma | — | $\geq 0$ | Measured |
| `comparator_offset_sigma__V` | static comparator-threshold sigma | V | $\geq 0$ | Measured |
| `comparator_thermal_noise_sigma__V` | per-cycle comparator-noise sigma | V | $\geq 0$ | Measured |
| `e_bootstrap__fJ` | per-conversion sampling-switch overhead | fJ | $\geq 0$ | Design |
| `e_compare_per_bit__fJ` | per-cycle comparator-decision energy | fJ | $\geq 0$ | Design |
| `e_logic_per_bit__fJ` | per-cycle SAR-logic / register overhead | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

The reference is not a config parameter - the family supplies the reference taps per call and the operating-point mode selects one (see [base](family.md)). Per-op latency is likewise not a parameter: it is derived as $(b+1)\cdot$ `clk_period__ns` from the operating point. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

Common symbols are in the family table in [base](family.md#symbols); this scheme adds:

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\Delta V_{\mathrm{top},k}$ | monotonic Set-and-Down top-plate step on cycle $k$ | V | derived |
| $V_{\mathrm{ref}}$ | reference voltage | V | `v_refs__V[..., mode]` |
| $C_k$ | capacitance of cap $k$ (mismatched after fabricate) | fF | `c_p__fF`, `c_n__fF` |
| $C_{\mathrm{total}}$ | total array capacitance, $2^{\,b_{\max}-1}C_{\mathrm{unit}}$ | fF | derived |
| $C_{\mathrm{unit}}$ | unit-cap capacitance | fF | `c_unit__fF` |
| $T$ | operating temperature | K | `T__K` |

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

# `neurox/analog/adc/sar_mono.py`

## Current role

`SarAdcMono` is the monotonic (Set-and-Down) differential SAR ADC. Fabrication state (per-leg cap arrays + comparator offset) is fully wired, but the differential convert kernel is currently a placeholder — calling `convert(...)` raises `NotImplementedError`.

For any current SAR work use [`mcs_sar.md`](mcs_sar.md) instead.

## Topology

Monotonic Set-and-Down switching:

- the larger top plate drops by `V_ref · C_k / C_total` each cycle while the smaller side holds
- caps only ever discharge to GND during the SAR loop; they never re-charge to `V_ref` once switched

The single-ended Set-and-Down variant has been retired in favour of the differential one planned here.

## Config

Same shape as `McsSarAdcConfig` (max_bits, `v_refs`, cap / comparator mismatch sigmas, energy overhead, PPA). Field names match the MCS SAR for forwards-compatibility.

## State that is wired today

`fabricate(shape)` already builds:

- `c_p__fF` / `c_n__fF` — independently-sampled per-leg cap arrays.
- `comparator_offset__V` — static threshold offset.

A future differential convert kernel will consume these plus the per-call kT/C noise and per-cycle comparator noise.

See also:

- `mcs_sar.md`
- `base.md`

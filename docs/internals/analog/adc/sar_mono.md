# Monotonic SAR ADC

## Summary

`SarAdcMono` (`adc/sar_mono.py`) is the monotonic (Set-and-Down) differential SAR ADC. Its fabrication state is wired but the differential conversion kernel is unrealised. Spec: [reference/analog/adc/sar_mono](../../../reference/analog/adc/sar_mono.md).

## Design decisions

- **Fabrication wired ahead of the kernel.** `_sample_fabricate_mismatch` already builds the per-leg cap arrays (`c_p__fF` / `c_n__fF`, independently sampled at `_inst_shape`) and the `comparator_offset__V`, so the static state path is exercisable before the convert kernel exists. The future kernel will consume these plus per-call kT/C and per-cycle comparator noise.
- **Config shape shared with the MCS variant.** Same fields as `McsSarAdcConfig` (`max_bits`, cap / comparator mismatch sigmas, energy overhead, static PPA) — including the same no-reference-ladder choice (the future kernel reads the injected `v_refs__V` tensor) and the same no-`latency_per_op__ns` choice (the future kernel will derive `(adc_operation_point.adc_bits + 1) * clk_period__ns`). The shared shape keeps the two SAR topologies swappable once the kernel lands.

## Contracts & invariants

- **`convert(...)` raises `NotImplementedError`.** The differential kernel is a placeholder; callers must not route production traffic here. Use [mcs_sar](mcs_sar.md) for current SAR work.
- **`SarAdcMonoPolicy` mirrors `McsSarAdcPolicy`** (`cap_mismatch`, `comparator_offset`, `comparator_thermal_noise`, `sampling_thermal_noise`).

## Performance & resources

N/A - no conversion path to profile yet.

## Gotchas

- **Do not treat `SarAdcMono` as a drop-in SAR.** The fabricated state is real but the conversion is not; selecting it as an operating ADC raises at `convert`.

## Known limitations

- **The differential convert kernel is unrealised.** Only the differential Set-and-Down variant is modelled; the single-ended variant is out of scope. Until the kernel lands the topology is fabrication-only and untested on the conversion path.

---

- **Reference**: [sar_mono](../../../reference/analog/adc/sar_mono.md)
- **Implementation**: `neurox/analog/adc/sar_mono.py`
- **Tests**: TODO - name the guarding test (none until the kernel is realised)
- **Decisions**: N/A — no ADR governs this module.

# Monotonic SAR ADC

`SarAdcMono` is a fabrication-only placeholder: the static-mismatch state is sampled, but the differential conversion kernel is unrealised.

## Design decisions

- **Fabrication wired ahead of the kernel.** `_sample_fabricate_mismatch` builds the two independently-sampled per-leg cap arrays (`c_p__fF`, `c_n__fF`) and `comparator_offset__V` at `_inst_shape`, so the static-mismatch path runs while the conversion kernel is still absent.

## Contracts & invariants

- **Policy and config stay in lockstep with the MCS SAR variant.** `SarAdcMonoPolicy` carries the same toggles as `McsSarAdcPolicy` (`cap_mismatch`, `comparator_offset`, `comparator_thermal_noise`, `sampling_thermal_noise`) and `SarAdcMonoConfig` reuses the MCS SAR field vocabulary; keeping them mirrored is what lets the two SAR topologies swap at the ADC interface.

## Known limitations

- **The differential convert kernel is unrealised**, so the conversion path is unverified.

---

- **Reference**: [sar_mono](../../../reference/analog/adc/sar_mono.md)
- **Implementation**: `neurox/analog/adc/sar_mono.py`
- **Tests**: TODO - name the guarding test (none until the kernel is realised)

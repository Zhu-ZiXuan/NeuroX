# Monotonic SAR voltage ADC

`SarMonoVoltageAdc` is a fabrication-only placeholder: the static-mismatch state is sampled, but the differential conversion kernel is unrealised.

## Design decisions

- **Fabrication wired ahead of the kernel.** `_sample_fabricate_mismatch` builds the two independently-sampled per-leg cap arrays (`c_p__fF`, `c_n__fF`) and `comparator_offset__V` at `_inst_shape`, so the static-mismatch path runs while the conversion kernel is still absent.

## Contracts & invariants

- **Policy and config stay in lockstep with the MCS SAR variant.** `SarMonoVoltageAdcPolicy` carries the same toggles as `McsSarVoltageAdcPolicy` (`cap_mismatch`, `comparator_offset`, `comparator_thermal_noise`, `sampling_thermal_noise`) and `SarMonoVoltageAdcConfig` reuses the MCS SAR field vocabulary; keeping them mirrored is what lets the two SAR topologies swap at the ADC interface.

## Known limitations

- **The differential convert kernel is unrealised**, so the conversion path is unverified.

---

- **Reference**: [sar_mono](../../../../reference/primitive/analog/voltage_adc/sar_mono.md)
- **Implementation**: `neurox/primitive/analog/voltage_adc/sar_mono.py`
- **Tests**: TODO - name the guarding test (none until the kernel is realised)

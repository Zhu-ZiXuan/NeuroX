# ADC family

The ADC family digitizes the readout chain's differential analog signal to a signed integer code. An abstract family contract plus its concrete topologies (a boundary-bucketize behavioural ADC and the SAR variants).

- [base](family.md) — the abstract `ADC` contract: signed-code convention, floor semantics, multi-mode operating point, what the ADC does and does not own.
- [general](general.md) — the boundary-bucketize behavioural ADC.
- [mcs_sar](mcs_sar.md) — the $V_{\mathrm{cm}}$-based (Merged Capacitor Switching) differential SAR ADC.
- [sar_mono](sar_mono.md) — the monotonic (Set-and-Down) differential SAR ADC.

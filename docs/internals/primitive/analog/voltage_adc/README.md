# Voltage ADC family

How the voltage ADC family is built.

- [base](base.md) — registry dispatch, the family init signature, signed-code zero-code placement, multi-mode and leaf-defined latency.
- [general](general.md) — the boundary-bucketize ADC and its cached zero code.
- [mcs_sar](mcs_sar.md) — the MCS SAR ADC, per-call zero code, compile considerations.
- [sar_mono](sar_mono.md) — the monotonic SAR ADC, wired state versus the unrealised kernel.

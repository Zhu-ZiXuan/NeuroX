# Current ADC family

The current ADC family digitizes a single-ended magnitude current into a raw unsigned integer code against a per-call reference ladder. The ladder has shape `[*R, n_ref]` with `n_ref = 2**bits - 1` taps ascending along the last axis; the `[*R]` leading broadcasts right-aligned against the input current, so each element quantizes against its own ladder. An abstract family contract plus its concrete topologies.

- [base](family.md) — the abstract `SingleEndedCurrentAdc` contract: single-ended magnitude input, per-call `[*R, n_ref]` reference ladder + `bits`, raw unsigned-code convention, `record_latency` gate on latency emission, what the ADC does and does not own.
- [sar](sar.md) — the triple-margin current-mode successive-approximation ADC.

# Voltage mux

## Design decisions

- **Single-ended N:1 transport.** `transport` accepts and preserves a
  caller-provided `(..., access_num, lane_num)` layout.
- **Connectivity belongs to the owner.** The owner maps source signals onto
  accesses and lanes before calling `transport`; the mux neither groups nor
  permutes axes.
- **Static mismatch vs dynamic noise.** `_sample_fabricate_mismatch` samples
  one fractional gain error per physical instance. `transport` applies
  additive voltage noise independently on every access.
- **Per-call PPA tally.** Dynamic energy is a flat per-access lump over the
  time-expanded output, emitted as a 0-dim constant expanded onto that layout,
  a view with no storage, so nothing is materialized. The mux reports no
  duration: the N inputs of a lane reach it one at a time on a schedule the
  owner runs, and the owner times it.
- **No instance axis is declared; access and lane fold together.** `lane_num`
  is the last `inst_shape` extent and `access_num == mux_ratio` sits to its
  left. The collector keeps only the caller's own leading dims and
  sums every axis past them — access, lane, and any outer instance axes alike
  — into the flat per-access constant, with nothing declared at the emission
  site.

## Contracts & invariants

- The two trailing axes must be `(mux_ratio, lane_num)`.
- `access_num` is serial and equals `mux_ratio`; `lane_num` is parallel and
  matches the final `inst_shape` extent.
- Gain mismatch is static across calls; transport noise is resampled per call.

## Performance & resources

The value tensor undergoes only elementwise gain and noise operations.

---

- **Reference**: [voltage_mux](../../../reference/primitive/analog/voltage_mux.md)
- **Implementation**: `neurox/primitive/analog/voltage_mux.py`
- **Tests**: `tests/primitive/analog/test_voltage_mux.py`

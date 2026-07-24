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
- **Per-call PPA tally.** Dynamic energy follows the time-expanded output.
  Latency is `latency_per_op * ceil(work_item_count / inst_count)`.

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

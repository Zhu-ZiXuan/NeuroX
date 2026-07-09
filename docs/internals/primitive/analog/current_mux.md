# Current mux

## Design decisions

- **Not polymorphic.** One concrete ideal mux, constructed directly rather than dispatched through a registry. It has no error sources, so its `Policy` is an empty marker.
- **Ideal value path, self-logged latency only.** The value path applies only the matched gain — no numeric non-ideality — and `transport` does not self-log rail energy. It still self-logs serial latency, so a lossless block contributes to the latency tally. Governing equations in Reference.
- **`select_num` is design fan-in, not a latency multiplier.** Latency scales by the runtime `serial_op_count` (`numel // inst_count`), never by `select_num`, which would double-count the already-tallied column visits (see Reference).

## Contracts & invariants

- **Canonical leaf signature.** The per-instance count is fixed from `inst_shape` and drives the serial-op count.

## Performance & resources

N/A — per-call elementwise map off the memory- and compile-critical path.

## Gotchas

- The latency accounting uses the runtime `serial_op_count`, never `select_num` — see Design decisions.

## Known limitations

- Ideal only: no policy hook exists to enable the non-idealities named in Reference.

---

- **Reference**: [current_mux](../../../reference/primitive/analog/current_mux.md)
- **Implementation**: `neurox/primitive/analog/current_mux.py`
- **Tests**: `tests/test_current_readout_energy.py`

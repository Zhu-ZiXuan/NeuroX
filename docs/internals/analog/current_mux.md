# Current mux

## Summary

`CurrentMux` (`current_mux.py`) is a leaf single-ended N:1 time-share current-transport block: it passes a selected lane current (scaled by `mux_gain`), self-logs the data-dependent rail conduction energy, and emits the time-share serial latency. Spec: [reference/analog/current_mux](../../reference/analog/current_mux.md).

## Design decisions

- **Not polymorphic.** One concrete ideal mux; constructed directly by the consuming circuit. No error sources yet, so its `Policy` is an empty marker.
- **Ideal value path, real energy.** `transport` is `i_out = mux_gain * i` (identity at unity gain; no on-resistance or charge-injection modelled). It self-logs `E = v_supply * |i_out| * read_pulse` (selected-lane rail conduction) and a latency `latency_per_op__ns * serial_op_count`. The input current is sourced externally (its production energy is accounted by the upstream block), so only the output lane is counted.
- **`select_num` is the design fan-in, NOT an energy/latency multiplier.** The `N` of the N:1 mux is a structural / area knob (the consuming tile cross-checks it `== ref_group_size`); the per-VMM access count is the runtime serial-op count (`numel // inst_count`), so neither the energy nor the latency is multiplied by `select_num` — doing so double-counts the fan-in.
- **`read_pulse__ns` is an `__init__` param** (one `readout_pulse__ns` source, like `T__K`), not a per-block config field.

## Contracts & invariants

- **Canonical leaf signature** plus `read_pulse__ns`. The per-instance count is locked from `inst_shape`.
- **Value-preserving** (identity at `mux_gain = 1`).

## Performance & resources

N/A — per-call elementwise map off the memory- and compile-critical path.

## Gotchas

- The energy / latency accounting uses the runtime `serial_op_count`, never `select_num` — see Design decisions.

## Known limitations

- Ideal only: on-resistance IR drop and charge injection are not modelled (no policy hooks yet).

---

- **Reference**: [current_mux](../../reference/analog/current_mux.md)
- **Implementation**: `neurox/analog/current_mux.py`
- **Tests**: `tests/test_current_readout_energy.py`
- **Decisions**: N/A — no ADR governs this module.

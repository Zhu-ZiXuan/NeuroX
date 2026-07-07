# Current mirror

## Summary

`CurrentMirror` (`current_mirror.py`) is a leaf single-ended current-copy block: it replicates a per-column current by a fixed `mirror_ratio` and self-logs the data-dependent rail energy it draws to do so.

## Design decisions

- **Not polymorphic.** One concrete mirror; the consuming circuit constructs it directly. Its `Policy` carries a single `mismatch` toggle.
- **Ratio-copy value path, real energy.** `replicate` is an `i_out = ratio * i_in` copy. With `mismatch` off the ratio is the exact `mirror_ratio`; with it on the ratio carries a static per-instance multiplicative Pelgrom Gaussian `1 + randn(inst_shape) * ratio_sigma_relative`, sampled once at `fabricate` into a held `ratio_mismatch` buffer and broadcast over the leading batch / im2col / element dims of `i_in`. The energy is data-dependent: only the replicated output current is drawn from the supply rail here — the input current is sourced externally and its production energy is accounted by the upstream block — so it self-logs `E = v_supply * |i_out| * read_pulse` on completion (naturally tracking the possibly-perturbed `i_out`). Self-logging keeps each block's rail draw from being silently dropped when blocks are composed in a chain.
- **`read_pulse__ns` is an `__init__` param, not a config field.** The read-window integration time is a single system quantity (one `readout_pulse__ns` source) injected at construction — like `T__K` / `dtype` — so it is never duplicated across block configs.

## Contracts & invariants

- **Canonical leaf signature** plus `read_pulse__ns`. The per-instance count is locked from `inst_shape` at construction.
- **Rail draw only.** The block models its own supply-drawn dynamic energy; static power lives in `leakage_per_inst__uW`.

## Performance & resources

N/A — `replicate` is a per-call elementwise map off the memory- and compile-critical path.

## Gotchas

- N/A.

## Known limitations

- Output compliance, finite output impedance, and input offset are not modelled. Copy-ratio mismatch is a static per-instance (Pelgrom) multiplicative Gaussian (`mismatch` toggle).

---

- **Reference**: [current_mirror](../../reference/analog/current_mirror.md)
- **Implementation**: `neurox/analog/current_mirror.py`
- **Tests**: `tests/test_current_readout_energy.py`

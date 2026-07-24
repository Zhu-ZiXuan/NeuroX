# Unmodeled block

## Design decisions

- **Static PPA seat only.** The block exposes no functional method — no `convert` / `transport` / `add`, no forward. It exists solely to carry `area__um2` and `leakage__uW` into the profiler's static walk. `_sample_fabricate_mismatch` is an explicit no-op: it owns no static mismatch state.
- **Reporter, not a roller-up.** Unlike the embedded non-reporter primitives, this block IS a profile target (`is_profile_target` left true): it sets the bare `_area_per_inst__um2` / `_leakage_per_inst__uW` in `__init__` from config, so its seat is reported directly ([base](base.md)).
- **No dynamic event.** The block never calls `_log_dynamic_energy`; its model covers static PPA only.
- **Empty policy.** No error sources; the `Policy` is an empty marker.

## Contracts & invariants

- **Canonical leaf signature.** The per-instance count is fixed from `inst_shape` at construction; static PPA scales by `inst_count`.
- **No signal-path presence.** With no functional method, the block is unreachable from any solve or convert; it only appears in the static report.

## Performance & resources

N/A — no runtime path.

## Known limitations

- The block's transfer and dynamic energy are not modelled; it contributes only to the static PPA tally.

---

- **Reference**: [unmodeled](../../../reference/primitive/analog/unmodeled.md)
- **Implementation**: `neurox/primitive/analog/unmodeled.py`
- **Tests**: `tests/primitive/analog/test_unmodeled.py`

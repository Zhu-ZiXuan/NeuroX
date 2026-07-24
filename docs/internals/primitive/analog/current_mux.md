# Current mux

## Design decisions

- **Not polymorphic.** One concrete ideal mux, constructed directly rather than dispatched through a registry. It has no error sources, so its `Policy` is an empty marker.
- **Ideal, lossless value path — no emission.** The value path applies only the matched gain and preserves the caller-provided `(..., access_num, lane_num)` layout. `transport` self-logs neither rail energy nor latency.
- **Connectivity belongs to the owner.** The owner maps source signals onto accesses and lanes before calling `transport`; the mux neither groups nor permutes axes.
- **The trailing axes distinguish serial and parallel work.** `access_num` is the number of sequential mux accesses and must equal `mux_ratio`; `lane_num` is the number of parallel output lanes and must match the final `inst_shape` extent.

## Contracts & invariants

- **Input layout.** The two trailing axes must be `(mux_ratio, lane_num)`.
- **Static PPA rolls up to the owner.** The mux's silicon is accounted in the owning current-domain circuit's config, so it declares no per-instance area or leakage data of its own and sets `is_profile_target` false ([base](base.md)).

## Performance & resources

N/A — the value path is one elementwise gain operation.

## Known limitations

- Ideal only: no policy hook exists to enable the non-idealities named in Reference.

---

- **Reference**: [current_mux](../../../reference/primitive/analog/current_mux.md)
- **Implementation**: `neurox/primitive/analog/current_mux.py`
- **Tests**: `tests/primitive/analog/test_current_mux.py`

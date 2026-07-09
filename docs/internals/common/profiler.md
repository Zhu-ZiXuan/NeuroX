# Profiler

## Summary

`NeuroxProfiler` (`common/profiler.py`) is the context manager that collects the PPA side channel: during its `with` block it captures the per-call dynamic energy / latency events that leaves emit, walks `CircuitBase` instances for static area / leakage, and aggregates both into a `ProfilerReport`. It is the collector half of the mechanism — the per-module emitter is the profile mixin — and how each PPA quantity is physically computed is a per-subsystem cost model.

## Design decisions

- **Energy / latency arrive on a side channel, not in return values.** Every numerical return in NeuroX is the analog / digital result; PPA quantities reach the profiler through the emission entries instead of being threaded back through call signatures. Rationale: the hot-path tensor pipeline stays narrow and a leaf deep in a composite can attribute its own cost without every intermediate function growing a PPA out-parameter. The cost is a discoverability one — emission sites are invisible from a function's signature.
- **Both quantities are collected as tensors, not Python floats.** Energy is genuinely per-element (per-instance switching energy summed by the profiler); latency is a per-op constant times a runtime serial-op count. Receiving latency as a 0-D tensor too lets it share the energy path's batched device→host sync instead of forcing a per-event `.item()` that would stall the pipeline on a GPU sync for one float.
- **One batched sync per quantity, deferred to context exit.** During the `with` block each emission stashes a 0-D reduction on the recording device; finalization does a single batched device→host transfer per quantity. Total host-sync cost is independent of event count. Rationale: a model with $10^5$ emitting calls would otherwise pay $10^5$ device syncs.
- **No static-walk cache.** `collect_static` re-walks the graph on every report rather than memoizing the result. The per-circuit area / leakage properties it reads are already cheap ([circuit](../primitive/circuit.md)), so a walk cache would add an invalidation surface for no measurable saving.
- **Static collection keys on `isinstance(m, CircuitBase)`, not `ProfileMixin`.** `CircuitBase` is the sole layer carrying the typed area / leakage surface, so a `ProfileMixin` host that is not a `CircuitBase` owns no static silicon and the walk skips it. The membership boundary — why a macro and a device fall outside `CircuitBase` — is in [circuit](../primitive/circuit.md).
- **`total_latency__ns` is a contribution sum, not wall-clock.** It sums every `LatencyEvent`, i.e. assumes sequential execution; a pipelined schedule would be shorter. It is the integration base for `ProfilerReport.leakage_energy__fJ = static.leakage_power__uW × total_latency__ns` — the single derivation point for leakage energy.

## Contracts & invariants

- **Record inside the `with`, read after it.** Inside the block each emission appends a 0-D tensor to a pending buffer; reading any aggregation or calling `report()` there sees the reset zeros, not partial state. `__exit__` clears the active-profiler slot, then finalizes **only on clean exit** — an exception leaves the partial state alone and avoids a stray sync that could mask the original error. After exit every accessor is a pure CPU field read.
- **Active profiler is thread-scoped.** The active profiler is held per thread; a leaf's emission is a no-op when no profiler is active on the calling thread. Emission and the enclosing `with` block must run on the same thread.
- **The profiler is the single source of aggregation.** Each dynamic contribution reaches the collector as its own event and the collector sums across events; it never re-sums, so the per-op totals are authoritative here and nowhere else. Folding a child's contribution into its parent, or emitting once per iteration of an internal chunked loop, would over-count — the emit-once emitter contract that holds each logical operation to one energy and one latency event is in [mixin/profile](mixin/profile.md). The sub-solvers and devices such a loop drives are not `CircuitBase` and emit nothing, so iteration alone never inflates the event count. This is the central correctness invariant of the side channel.
- **The serial-op count belongs to the leaf, not the collector.** A leaf hands over a latency tensor already scaled by its serial-op count — generically `max(1, numel // inst_count)`, output elements over the parallel fabrication multiplicity `inst_count` ([circuit](../primitive/circuit.md)), or a leaf-specific override. That derivation is part of the leaf's own cost model, in its `reference/<subsystem>/` page; the collector only requires the tensor.

### Public API

- `total_dynamic_energy__fJ` / `total_latency__ns` — pre-computed scalar sums over all `EnergyEvent` / `LatencyEvent`.
- `energy_by_name` / `energy_by_type` / `latency_by_name` — pre-computed grouped sums (`dict[str, float]`) keyed by emitter qualified name or module type.
- `energy_events: list[EnergyEvent]` / `latency_events: list[LatencyEvent]` — the resolved per-call event lists, populated by finalization.
- `analyze_static(model) -> StaticMetrics` — static-only staticmethod; aggregates the `CircuitBase` walk to totals. `StaticMetrics` carries area + leakage only (`area__um2`, `leakage_power__uW`), no dynamic quantity.
- `collect_static(model) -> list[StaticRecord]` — static-only staticmethod; the per-module records behind `analyze_static`. Each `StaticRecord` carries `qualified_name`, `module_type`, `area__um2`, `leakage_power__uW`.
- `report(model) -> ProfilerReport` — bundles this context's runtime events with a fresh static walk of `model`; must be called after the block exits. `ProfilerReport` exposes `total_dynamic_energy__fJ` / `total_latency__ns` over its event lists and the derived `leakage_energy__fJ`. `analyze_model(model)` is the static-only sibling (a report with no runtime events).
- `summary(*, static=None, extras=None) -> str` — concise multi-line text dump of the dynamic totals; when `static` is supplied it appends area, leakage power, and the derived leakage energy.

## Performance & resources

Per emission: one 0-D reduction (a kernel launch, no host sync) plus a Python append. Finalization does one batched device→host transfer per quantity — two host syncs total, independent of event count. Static collection is one `model.modules()` walk per `report` / `collect_static` / `analyze_static` call with $O(\mathrm{circuits})$ derived-property reads, no allocation of node-level tensors.

## Gotchas

- **Do not read totals mid-`with`.** The pending buffers hold device tensors that have not been drained; the aggregation accessors still read their entry-reset zeros. The values are not "partial" — they are stale.
- **An exception suppresses finalization.** If the `with` body raises, no sync runs and the event lists stay empty by design. Do not rely on profiler state after a failed run.
- **`latency_by_type` does not exist.** Latency is grouped by name only; energy alone has both `_by_name` and `_by_type`.

## Known limitations

- **No tensor-return profiling path.** A return-value channel that removes the emission-site graph break is an open option (tracked in [compile](../compile/README.md)); the side-channel break is accepted for now.
- **No CSV / JSON export, no benchmark suite.** Report consumption is in-process (`ProfilerReport`, `summary`) only.
- **Verification covers events and gating, not sync micro-cost.** `tests/test_xbar_chunking.py` guards single-event-per-operation and chunk-invariance; `tests/test_readout_log_gating.py` guards independent energy / latency gating. The batched-sync cost claim (one host sync per quantity) is a design invariant, not a regression-tested one.

---

- **Reference**: N/A — the profiler has no physics spec; per-subsystem PPA cost models live under [reference/](../../reference/README.md)
- **Implementation**: `neurox/common/profiler.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_readout_log_gating.py`

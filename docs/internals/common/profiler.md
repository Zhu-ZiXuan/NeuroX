# Profiler — Implementation

## Summary

`NeuroxProfiler` (`common/profiler.py`) is the context manager that collects the PPA side channel: during its `with` block it captures the per-call dynamic energy / latency events that leaves emit, walks `CircuitBase` instances for static area / leakage, and aggregates both into a `ProfilerReport`. This document covers only the **collector** — how energy / latency / area / leakage travel from a leaf into a report and the public accessors over that report. The emission side (`ProfileMixin`, the `_log_dynamic_energy` / `_log_latency` entries) is documented in [mixin/profile](mixin/profile.md). The PPA *cost model* — how each quantity is physically computed — is per-subsystem and lives in `reference/<subsystem>/` (e.g. [reference/xbar/_1t1r/core](../../reference/xbar/_1t1r/core.md) §Energy model, [reference/digital/accumulator](../../reference/digital/accumulator.md) §PPA cost model); it is not repeated here.

## Design decisions

- **Energy / latency arrive on a side channel, not in return values.** Every numerical return in NeuroX is the analog / digital result; PPA quantities reach the profiler through the emission entries instead of being threaded back through call signatures. Rationale: the hot-path tensor pipeline stays narrow and a leaf deep in a composite can attribute its own cost without every intermediate function growing a PPA out-parameter. The cost is a discoverability one — emission sites are invisible from a function's signature.
- **Both quantities are collected as tensors, not Python floats.** Energy is genuinely per-element (per-instance switching energy summed by the profiler); latency is a per-op constant times a runtime serial-op count. Receiving latency as a 0-D tensor too lets it share the energy path's batched device→host sync instead of forcing a per-event `.item()` that would stall the pipeline on a GPU sync for one float.
- **One batched sync per quantity, deferred to context exit.** During the `with` block each emission stashes a 0-D reduction on the recording device; finalization does a single batched device→host transfer per quantity. Total host-sync cost is independent of event count. Rationale: a model with $10^5$ emitting calls would otherwise pay $10^5$ device syncs.
- **No static-PPA cache.** `inst_area__um2` / `inst_leakage__uW` are `area_per_inst × inst_count`, read on access. `collect_static` walks the graph at most once per report, so a cache would add invalidation surface for no measurable saving.
- **`isinstance(m, CircuitBase)` is the static-collection predicate, not `ProfileMixin`.** `CircuitBase` is the layer that carries the typed area / leakage surface; a macro (a `ProfileMixin` but not a `CircuitBase`) has a name and may emit dynamic events but owns no static silicon, so it must not appear in the static report. Devices are not `CircuitBase` and roll up into the owning circuit's config — see [`ADR-0002`](../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md).
- **`total_latency__ns` is a contribution sum, not wall-clock.** It sums every `LatencyEvent`, i.e. assumes sequential execution; a pipelined schedule would be shorter. It is the integration base for `ProfilerReport.leakage_energy__fJ = static.leakage_power__uW × total_latency__ns` — the single derivation point for leakage energy.

## Contracts & invariants

- **Record inside the `with`, read after it.** Inside the block each emission appends a 0-D tensor to a pending buffer; reading any aggregation or calling `report()` there sees the reset zeros, not partial state. `__exit__` clears the active-profiler slot, then finalizes **only on clean exit** — an exception leaves the partial state alone and avoids a stray sync that could mask the original error. After exit every accessor is a pure CPU field read.
- **Active profiler is thread-scoped.** The active profiler is held per thread; a leaf's emission is a no-op when no profiler is active on the calling thread. Emission and the enclosing `with` block must run on the same thread.
- **Composites do not sum children's emissions into their own.** A composite emits only its own per-op overhead; each child with a dynamic model emits its own contribution directly. The profiler-level event aggregation is the single source of truth, so summing children at the composite would double-count. This is the central correctness invariant of the side channel.
- **One energy event + one latency event per logical operation.** A plain leaf emits inline once at the end of forward. A composite whose body contains an internal chunked / iterated loop (currently only `Core1T1R.solve_array`) must still emit exactly once per VMM: aggregate per-chunk inside the loop, then make the two emit calls once after it. Sub-solvers and devices inside the loop are not `CircuitBase` and emit nothing, so there is no double-count. A concrete scheme's chunking test covers the single-event invariant.
- **`ReadOut.readout` is the one serial-op rule that differs from generic `inst_count`.** The generic leaf rule is `serial = max(1, output.numel() // parallel_count)` with `parallel_count = inst_count`. `ReadOut.readout` output carries an extra parallel `slice_num` trailing dim (each slice position is a physically parallel readout path), so its `parallel_count = inst_count * slice_num`. Every other emitting leaf — `Adder` / `Subtractor` / `Accumulator` / `ShiftAdder`, `DAC.convert` / `ADC.convert` / `VoltageMux.transport`, `SwitchCap.sample_and_accumulate` — uses the plain `inst_count`. The full per-leaf serial-op derivation is each leaf's cost model and lives in its `reference/<subsystem>/` page.

### Public API

The runtime accessors below are populated by finalization (auto-run by `__exit__` on clean context exit) and are pure CPU field reads thereafter; reading them inside the `with` block sees the entry-reset zeros, not partial state.

- `total_dynamic_energy__fJ` / `total_latency__ns` — pre-computed scalar sums over all `EnergyEvent` / `LatencyEvent`.
- `energy_by_name` / `energy_by_type` / `latency_by_name` — pre-computed grouped sums (`dict[str, float]`) keyed by emitter qualified name or module type. Latency has no `_by_type` group.
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
- **The serial-op count, not the cost model, is the leaf's shape responsibility.** The collector only requires a tensor; how a leaf derives its serial-op multiplicity (numel / parallel-count, or the A-position product for the array core) is part of each leaf's cost model and is specified in its `reference/<subsystem>/` page, not here.

## Known limitations

- **No tensor-return profiling path.** A return-value channel that removes the emission-site graph break is an open option (tracked in [compile](../compile/README.md)); the side-channel break is accepted for now.
- **No CSV / JSON export, no benchmark suite.** Report consumption is in-process (`ProfilerReport`, `summary`) only.
- **Verification covers events and gating, not sync micro-cost.** `tests/test_xbar_chunking.py` guards single-event-per-VMM and chunk-invariance; `tests/test_readout_log_gating.py` guards independent energy / latency gating. The batched-sync cost claim (one host sync per quantity) is a design invariant, not a regression-tested one.

---

- **Reference**: N/A — the profiler has no physics spec; per-subsystem PPA cost models live under [reference/](../../reference/README.md) (e.g. [reference/xbar/_1t1r/core](../../reference/xbar/_1t1r/core.md), [reference/digital/accumulator](../../reference/digital/accumulator.md))
- **Implementation**: `neurox/common/profiler.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_readout_log_gating.py`
- **Decisions**: [`ADR-0002`](../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)

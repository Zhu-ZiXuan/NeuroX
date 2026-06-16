# Profiler — Implementation

## Summary

The PPA collection mechanism: `ProfileMixin` (`common/mixin/profile.py`) gives every profiled module a hierarchical name plus two side-channel log entries; `NeuroxProfiler` (`common/profiler.py`) is the context manager that captures those events, walks `CircuitBase` instances for static PPA, and aggregates both into a `ProfilerReport`. This document covers only the **collection mechanism** — how energy / latency / area / leakage travel from a leaf to a report. The PPA *cost model* (how each quantity is physically computed) is per-subsystem and lives in `reference/<subsystem>/` (e.g. [reference/xbar/_1t1r/circuit_core](../reference/xbar/_1t1r/circuit_core.md) §PPA, [reference/digital/accumulator](../reference/digital/accumulator.md) §PPA cost model); it is not repeated here.

## Design decisions

- **Energy / latency travel a side channel, not the return value.** Every numerical return in NeuroX is the analog / digital result; PPA quantities are pushed through `_log_dynamic_energy` / `_log_latency` instead of being threaded back through call signatures. Rationale: the hot-path tensor pipeline stays narrow and a leaf deep in a composite can attribute its own cost without every intermediate function growing a PPA out-parameter. The cost is a discoverability one — emission sites are invisible from a function's signature.
- **Two independent entries, never one fused PPA call.** `_log_dynamic_energy` and `_log_latency` are separate methods because a leaf may own a model for one quantity and not the other (a latency-only orch, an energy-only block). A single `log(energy, latency)` entry would force every caller to fabricate a zero for the quantity it does not model, and a single `if cost > 0` gate would silently drop the latency-only case. Each call site gates each quantity on its own. Rejected: a fused entry; verified against by `tests/test_readout_log_gating.py`, which covers all four (energy, latency) on/off corners.
- **No value-based gating inside the log methods.** The methods route a tensor and nothing else; the "should I emit?" decision is the leaf's, made at the call site from its config. This keeps the mixin physics-agnostic — it carries no PPA fields and no thresholds.
- **Both quantities are tensors, not Python floats.** Energy is genuinely per-element (per-instance switching energy summed by the profiler); latency is a per-op constant times a runtime serial-op count. Packaging latency as a 0-D tensor too lets it share the energy path's `.detach().sum()` → batched GPU→CPU sync, instead of forcing a per-event `.item()` that would stall the pipeline on a GPU sync for one float.
- **One batched sync per quantity, deferred to context exit.** During the `with` block each emission stashes a `.detach().sum()` 0-D tensor on the recording device; `_finalize` does a single `torch.stack → .cpu().tolist()` per quantity. Total host-sync cost is independent of event count. Rationale: a model with $10^5$ emitting calls would otherwise pay $10^5$ device syncs.
- **No static-PPA cache.** `inst_area__um2` / `inst_leakage__uW` are `area_per_inst × inst_count`, computed on access. `collect_static` walks the graph at most once per report, so a cache would add invalidation surface for no measurable saving.
- **`isinstance(m, CircuitBase)` is the static-collection predicate, not `ProfileMixin`.** `CircuitBase` is the layer that carries the typed area / leakage surface; a pure `ProfileMixin` subclass (a macro) has a name and may emit dynamic events but owns no static silicon, so it must not appear in the static report. Devices are not `CircuitBase` and roll up into the owning circuit's config — see [`ADR-0002`](../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md).

## Contracts & invariants

- **`@torch.compiler.disable` on both log methods.** The profiler reads `threading.local` and mutates a Python list — both untraceable by dynamo — so `_log_dynamic_energy` / `_log_latency` must stay outside the compiled graph. The break is local: it lands at the end of a primary method after all kernel math, so in-kernel fusion is unaffected. Library code therefore cannot assume `fullgraph=True`. See [compile/contracts](compile/contracts.md).
- **Composites do not sum children's emissions into their own.** A composite emits only its own per-op overhead (`self.config.*`); each child with a dynamic model emits its own contribution directly. The profiler-level event aggregation is the single source of truth, so summing children at the composite would double-count. This is the central correctness invariant of the side channel.
- **One energy event + one latency event per logical operation.** A plain leaf emits inline once at the end of forward. A composite whose body contains an internal chunked / iterated loop (currently only `CircuitCore1T1R.cim_read`) must still emit exactly once per VMM: aggregate per-chunk inside the loop, then make the two log calls once after it. Sub-solvers and devices inside the loop are not `CircuitBase` and emit nothing, so there is no double-count. Verified by `tests/test_xbar_chunking.py::test_profiler_single_event_under_chunking`.
- **Record inside the `with`, read after it.** Inside the block each `_log_*` appends a 0-D tensor to a pending buffer; reading any aggregation or calling `report()` there sees stale / unsynced state. `__exit__` clears the active-profiler slot, then calls `_finalize` **only on clean exit** (an exception leaves the partial state alone and avoids a stray sync that could mask the original error). After exit every property is a pure CPU field read.
- **Active profiler is thread-scoped.** `get_current` reads `threading.local`; a leaf's `_log_*` is a no-op when no profiler is active on the calling thread. Emission and the enclosing `with` block must run on the same thread.
- **`total_latency__ns` is a contribution sum, not wall-clock.** It sums every `LatencyEvent`, i.e. assumes sequential execution; a pipelined schedule would be shorter. It is the integration base for `ProfilerReport.leakage_energy__fJ = static.leakage_power__uW × total_latency__ns` — the single derivation point for leakage energy.
- **`ReadOut.readout` is the one serial-op rule that differs from generic `inst_count`.** The generic leaf rule is `serial = max(1, output.numel() // parallel_count)` with `parallel_count = inst_count`. `ReadOut.readout` output carries an extra parallel `data_num` trailing dim (each data position is a physically parallel readout path), so its `parallel_count = inst_count * data_num`. Every other emitting leaf — `Adder` / `Subtractor` / `Accumulator` / `ShiftAdder`, `DAC.convert` / `ADC.convert` / `AnalogMux.transport`, `SwitchCap.sample_and_accumulate` — uses the plain `inst_count`. The full per-leaf serial-op derivation is each leaf's cost model and lives in its `reference/<subsystem>/` page.

### Public API

The accessors below are populated by `_finalize` (auto-invoked by `__exit__` on clean context exit) and are pure CPU field reads thereafter; reading them inside the `with` block sees the reset zeros, not partial state.

- `total_dynamic_energy__fJ` / `total_latency__ns` — pre-computed scalar sums over all `EnergyEvent` / `LatencyEvent`.
- `energy_by_name` / `energy_by_type` / `latency_by_name` — pre-computed grouped sums (`dict[str, float]`) keyed by emitter name or module type.
- `analyze_static(model) -> StaticMetrics` — aggregate the static walk to totals; `StaticMetrics` carries area + leakage only (`area__um2`, `leakage_power__uW`), no dynamic quantity.

## Performance & resources

Per emission: one `.detach().sum()` (a kernel launch, no host sync) plus a Python tuple append. Per `_finalize`: one `torch.stack` + one `.cpu().tolist()` per quantity — two host syncs total, independent of event count. Static collection is one `model.modules()` walk per `report()` / `collect_static` call with $O(\text{circuits})$ derived-property reads, no allocation of node-level tensors.

## Gotchas

- **A graph break is expected at every emit site.** The `@torch.compiler.disable` boundary means a leaf's primary method cannot compile as one `fullgraph` region. Treat the break as designed; set `fullgraph=True` only in tests that want to catch an *accidental* Python sync elsewhere.
- **Do not read totals mid-`with`.** The pending buffers hold device tensors that have not been drained; the aggregation properties still read their `__enter__`-reset zeros. The values are not "partial" — they are stale.
- **An exception suppresses `_finalize`.** If the `with` body raises, no sync runs and the event lists stay empty by design. Do not rely on profiler state after a failed run.
- **Latency is not children-summed at composites.** A composite's `latency_per_op__ns` is its own glue overhead only. Reading it as "this subtree's latency" is wrong; the subtree total is the event-stream sum over all emitters, which the profiler provides.
- **The serial-op count, not the cost model, is the leaf's shape responsibility here.** The mechanism only requires a tensor; how a leaf derives its serial-op multiplicity (numel / parallel-count, or the A-position product for the array core) is part of each leaf's cost model and is specified in its `reference/<subsystem>/` page, not here.

## Known limitations

- **No tensor-return profiling path.** A return-value channel that removes the single graph break is an open option (tracked in [compile](compile/README.md)); the side-channel break is accepted for now.
- **No CSV / JSON export, no benchmark suite.** Report consumption is in-process (`ProfilerReport`, `summary`) only.
- **Verification covers events and gating, not sync micro-cost.** `tests/test_xbar_chunking.py` guards single-event-per-VMM and chunk-invariance; `tests/test_readout_log_gating.py` guards the four-corner independent gating. The batched-sync cost claim (one host sync per quantity) is a design invariant, not a regression-tested one.

---

- **Reference**: per-subsystem PPA cost models under [reference/](../reference/README.md) (e.g. [reference/xbar/_1t1r/circuit_core](../reference/xbar/_1t1r/circuit_core.md), [reference/digital/accumulator](../reference/digital/accumulator.md)) — the profiler itself has no physics spec
- **Implementation**: `neurox/common/mixin/profile.py`, `neurox/common/profiler.py`, `neurox/common/circuit.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_readout_log_gating.py`
- **Decisions**: [`ADR-0002`](../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)

# Profiler Modules

`neurox/profiler/` collects the per-module PPA / energy / latency side-channel.

Two pieces:

- `ProfiledModule` — mixin that gives a module a hierarchical profiler name and exposes `_log_dynamic(energy, latency)`. Every analog and digital leaf inherits this mixin to emit per-call dynamic events.
- `NeuroxProfiler` — context manager that intercepts the events, groups them by hierarchical name, and produces per-layer PPA reports.

## Hierarchical naming

A profiled module receives a hierarchical name from its constructor (e.g. `<layer>.<owner>.<member>` — `fc1.foo.bar.baz`), and each child circuit appends its own role to that name when it constructs further children. Devices do **not** carry profiler names — their PPA rolls up to the owning circuit.

## Side-channel only

Every numerical return value in NeuroX is the actual analog / digital result; energy and latency travel through the profiler side channel only. That keeps the hot-path tensor pipeline narrow and `@torch.compile`-clean.

## Static vs dynamic

Static PPA (area, leakage power, leakage energy, latency) is reported through the same profiler interface but aggregated once per session rather than per call. Each leaf circuit exposes the static fields via the canonical `area_per_inst__um2` / `leakage_per_inst__uW` / `latency_per_op__ns` properties; the profiler aggregates them by hierarchical name when a report is requested.

See also:

- `docs/dev/roadmap.md` (CSV / JSON export and benchmark suite plans)

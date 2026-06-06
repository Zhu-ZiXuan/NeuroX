# Profiler and PPA Accounting

This document records the rules for static area / leakage reporting, dynamic energy logging, and the `ProfileMixin` side channel.

## Who profiles, who does not

Every **circuit module** (anything under `neurox/analog/`, `neurox/digital/`, `neurox/xbar/`, `neurox/macro/`) must inherit `ProfileMixin`. This is non-negotiable.

**Device modules** (`neurox/device/`) do **not** inherit `ProfileMixin`. Device PPA rolls up to the owning circuit — see [`ADR-0002`](docs/dev/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md). Device modules expose no `_log_dynamic` / `area_per_inst__um2` / `leakage_per_inst__uW` / `latency_per_op__ns`.

## Required interface

Every `ProfileMixin` exposes:

- `area_per_inst__um2: float` — property. Silicon area of a single instance of this module.
- `leakage_per_inst__uW: float` — property. Static leakage of a single instance.
- `latency_per_op__ns` — property when the per-op latency is fixed; method when it depends on a runtime operating point. The method form takes the operating point by keyword (e.g. `latency_per_op__ns(*, adc_operation_point: AdcOperationPoint) -> float`).

## `_log_static` and the `_inst_shape` contract

Every `ProfileMixin` records its static PPA via `_log_static()` — no argument; reads `self._inst_shape` and the subclass's `area_per_inst__um2 / leakage_per_inst__uW` properties to populate `_inst_count / _inst_area__um2 / _inst_leakage__uW`.

```python
def __init__(self, *, config, name, inst_shape, dtype, T__K) -> None:
    super().__init__(...)              # ProfileMixin.__init__ stores name
    self._inst_shape = inst_shape      # FabricateMixin + ProfileMixin contract
    self.config = config                     # subclass-specific
    # ... register buffers / derive scalars ...
    self._log_static()                  # only the most-derived concrete subclass calls
```

Rules:

- `_log_static()` is called **exactly once** at the end of the most-derived concrete subclass's `__init__`. Family bases (e.g. `ADC`, `DAC`, `TIA`, `ReadOut`, `Xbar`, `XbarMacro`) do **not** call it themselves — avoids double-recording.
- Required ordering: `self.config = config` and `self._inst_shape = inst_shape` must be set before the call, because the property pair `area_per_inst__um2 / leakage_per_inst__uW` reads `self.config.*`.
- Modules with no own static PPA (currently the `XbarMacro` family) override `area_per_inst__um2 = 0.0` / `leakage_per_inst__uW = 0.0` properties and still call `_log_static()`, contributing 0 to the static rollup.
- The fields `_inst_count / _inst_area__um2 / _inst_leakage__uW` are **not pre-initialised** on `ProfileMixin`. Forgetting `_log_static()` surfaces as `AttributeError` on the first profiler walk, not a silent zero.

`_inst_shape` is the **circuit-instance count shape**: each cell of a tensor of this shape represents one independent fabricated instance of the module. Multi-dimensional to mirror the tile / batch structure; the profiler treats it as a count via `math.prod(_inst_shape)`.

## Composite-module PPA aggregation

When a composite circuit owns child circuit modules:

- Composite's `area_per_inst__um2` and `leakage_per_inst__uW` return **only the composite's own extra cost** (`self.config.area_per_inst__um2` etc.). For `XbarMacro` and similar pure orchestration nodes that own no silicon themselves, both properties return `0.0`.
- Each child records its own instance count via its own `_log_static()` at construction (the child's `inst_shape` was bound by the composite's `__init__`).
- The profiler walks `model.modules()` and sums the `inst_area__um2` / `inst_leakage__uW` of every `ProfileMixin` it finds. Composite and children both contribute their own shares; nothing is double-counted.

Latency aggregation is different: `latency_per_op__ns` on a composite typically returns the **pipeline sum** of children's per-op latency plus the composite's own contribution. Pipeline depth is not a sum of instance counts; the two aggregation modes are by design different.

## Dynamic energy and the side channel

Primary methods (the ones listed under "primary-method names" in [`naming_conventions.md`](naming_conventions.md)) emit dynamic energy by calling `self._log_dynamic(dynamic_energy__fJ, latency__ns)` at the end of the method.

Rules:

- `dynamic_energy__fJ` is a **tensor** in `__fJ`. The profiler sums it internally via `.sum().item()`.
- `latency__ns` is a Python `float` in `__ns`.
- One primary method should call `_log_dynamic` at most once. Helper methods invoked from the primary do not log on their own behalf.
- Composite modules log only their own extra dynamic energy (e.g. orchestrator-level overhead). Children's energy enters the profiler via their own `_log_dynamic` calls.

## Why `_log_dynamic` is decorated `@torch.compiler.disable`

`_log_dynamic` is the **only intentional graph break** on the runtime path. It is necessary because:

1. `NeuroxProfiler.get_current()` reads `threading.local()` — dynamo cannot trace thread-local state.
2. `profiler._append_runtime_event(...)` mutates a Python list — dynamo cannot trace host-state mutation.

The decorator localises the break to one well-defined call at the end of each primary method. The break happens between primary methods, not inside the inner loop of any kernel, so `torch.compile` fusion within each kernel is unaffected.

`_log_dynamic` does take `.sum().item()` on the energy tensor inside the disabled region. Just `.sum()` alone (without `.item()`) does not avoid the break — Python state mutation is what forces it, not CPU sync. Avoiding the break entirely would require a redesign that returns energy via the call's tensor return path; the current side-channel is a deliberate trade-off and not a target for optimisation.

## Profiler entry points

`NeuroxProfiler` is a context manager. Within its `with` block, every `_log_dynamic` call appends to the active profiler's event list; outside the block the calls are no-ops. The profiler exposes:

- `events` — runtime event list, in call order.
- `analyze_static(model) -> StaticMetrics` — walks `model.modules()` and sums area / leakage.
- `analyze_model(model) -> ProfilerReport` — static-only report (no runtime events).
- `report(model) -> ProfilerReport` — bundles the runtime events with the static aggregation.

`StaticMetrics.leakage_energy__fJ = leakage_power__uW · total_latency__ns` is computed once, centrally, in `summary(...)`.

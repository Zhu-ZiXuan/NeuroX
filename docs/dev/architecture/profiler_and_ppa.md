# Profiler and PPA Accounting

This document records the rules for static area / leakage reporting, dynamic energy + latency logging, and the `CircuitBase` / `ProfileMixin` split.

## Who profiles, who does not

Every **electrical circuit module** (anything under `neurox/analog/`, `neurox/digital/`, `neurox/xbar/`) inherits `CircuitBase[ConfigT]`, which composes `FabricateMixin + ProfileMixin + nn.Module + Generic[ConfigT]` and exposes the typed static-PPA surface.

**Macro modules** (`neurox/macro/`) are orchestration nodes; they own zero silicon themselves and dispatch into children. They inherit `FabricateMixin + nn.Module + ProfileMixin + RegistryMixin` directly (no `CircuitBase`) and their PPA appears in reports through their constituent circuits.

**Device modules** (`neurox/device/`) do **not** inherit `CircuitBase` and do **not** emit profile events. Device PPA rolls up to the owning circuit — see [`ADR-0002`](../adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md). Their `*Config` does not inherit `CircuitConfig`.

## Required interface

Every `CircuitConfig` carries two static-PPA fields:

```python
@dataclass(frozen=True)
class CircuitConfig(ValidateMixin):
    area_per_inst__um2: float
    leakage_per_inst__uW: float
```

Every `CircuitBase` exposes five properties:

- `area_per_inst__um2` / `leakage_per_inst__uW` — read from `self.config`
- `inst_shape` — forwarder over the bound `_inst_shape` tuple
- `inst_count` — `math.prod(self._inst_shape)`
- `inst_area__um2` / `inst_leakage__uW` — per-instance × inst_count, computed on access

Per-op latency is **not** a base contract — see the next section.

## Where per-op latency comes from

`_log_latency(latency)` takes a latency tensor that the leaf assembles itself; there is no `CircuitBase.latency_per_op__ns` property. Two patterns:

- **Fixed-latency emitting leaves** (`SwitchCap`, `AnalogMux`, `GeneralDAC`, `GeneralADC`, `Adder`, `Subtractor`, `Accumulator`, `ShiftAdder`, `CircuitCore1T1R`, `OffsetSwitchCapMuxAdcReadOut`) declare `latency_per_op__ns: float` on their *own* `*Config`, validate it in `validate_ppa`, and read `self.config.latency_per_op__ns` at the emit site.
- **Parametric-latency emitting leaves** (`McsSarAdc`, future `SarAdcMono`, …) do **not** declare a `latency_per_op__ns` field; their `convert(...)` body computes per-op latency from runtime knobs — e.g. `(adc_operation_point.adc_bits + 1) × config.clk_period__ns` for an SAR ADC — and feeds the resulting tensor into `_log_latency`. There is no placeholder field on the config.
- **Non-emitting leaves** (`Driver`, `OpAmpTIA`) are still `CircuitBase` subclasses carrying static PPA, but their primary methods do not call `_log_dynamic_energy` / `_log_latency` — their physical contribution is folded into the owning composite's energy / latency tensors (e.g. `CircuitCore1T1R._compute_array_energy__fJ` covers TIA / Driver / RRAM / NMOS jointly).

Latency and dynamic energy occupy the same conceptual slot in the side-channel emission — both are runtime quantities packaged into tensors and sunk through the same per-quantity batched sync, but they are **independent**: each leaf decides whether to emit one, the other, both, or neither.

## No PPA cache

`inst_area__um2` and `inst_leakage__uW` are derived properties — `math.prod(_inst_shape) × per-instance` is computed on access. The profiler walks the model graph at most once per report, so a cache adds no value.

A leaf circuit therefore writes no PPA-bookkeeping call at the end of its `__init__`:

```python
class Driver(CircuitBase[DriverConfig]):
    def __init__(self, *, config, policy, name, inst_shape, dtype, T__K) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K
        self.register_buffer("nominal_drive_value", ..., persistent=False)
```

## `_log_dynamic_energy` / `_log_latency` — two independent entries

`ProfileMixin` exposes two side-channel emission methods, one per quantity:

```python
def _log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None: ...
def _log_latency(self, latency__ns: Tensor) -> None: ...
```

Each takes a tensor and performs no value-based gating — any "should I emit?" decision lives at the call site. Caller-side construction in a fixed-latency leaf that emits both:

```python
def some_op(self, ...):
    y = ...
    op_count = self._serial_op_count(y)                              # leaf-specific
    dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ)
    latency__ns        = torch.tensor(self.config.latency_per_op__ns * op_count,
                                      device=y.device, dtype=dynamic_energy__fJ.dtype)
    self._log_dynamic_energy(dynamic_energy__fJ)
    self._log_latency(latency__ns)
```

In a parametric-latency leaf (e.g. `McsSarAdc.convert`) the only difference is how `per_op_latency__ns` is computed:

```python
per_op_latency__ns = (adc_operation_point.adc_bits + 1) * config.clk_period__ns
latency__ns        = torch.tensor(per_op_latency__ns * op_count, device=..., dtype=...)
self._log_dynamic_energy(dynamic_energy__fJ)
self._log_latency(latency__ns)
```

Composite-orch leaves (e.g. `OffsetSwitchCapMuxAdcReadOut`) gate each quantity independently — a latency-only orch records its latency event even when `energy_per_op__fJ` is 0:

```python
if self.config.energy_per_op__fJ > 0.0:
    self._log_dynamic_energy(...)
if self.config.latency_per_op__ns > 0.0:
    self._log_latency(...)
```

Why both are tensors:

- **Energy** is per-call dynamic data; building it as a tensor lets each element record per-instance switching energy that the profiler sums.
- **Latency** is per-op static data multiplied by a runtime serial-op count. By packaging it as a tensor too, the profiler treats it with the same `.detach().sum()` → batched GPU→CPU sync flow as energy — no special-case path for one constant float per event, no per-call `.item()` that would force a GPU sync.

The result: at `_finalize` (auto-invoked by `__exit__` on clean exit) the profiler does **two** batched syncs — one for the energy list, one for the latency list — and writes float values into `EnergyEvent.dynamic_energy__fJ` / `LatencyEvent.latency__ns`. Total cost is independent of event count.

## Serial-op count, per circuit

`latency = per_op_latency × op_count`. The `op_count` is the **serial** invocation count for this forward call — the dim-tally of the input tensor minus the dims that are physically parallel (the inst dims plus any per-circuit trailing dims like `n_caps` on a switch-cap bank or `digit_num` on a readout).

Every emitting leaf uses the position-invariant numel rule
``serial = max(1, output.numel() // parallel_count)`` where
``parallel_count`` reflects the leaf's parallel-hardware multiplicity
in its output tensor — `inst_count` for most leaves; with extra
factors when the output has additional parallel trailing:

- `Adder` / `Subtractor` / `Accumulator` / `ShiftAdder`: `parallel = inst_count`
- `DAC.convert` / `ADC.convert` / `AnalogMux.transport`: `parallel = inst_count`
- `SwitchCap.sample_and_accumulate`: output is the post-reduce `v_out` (`n_caps` already gone), so `parallel = inst_count`
- `ReadOut.readout`: output carries an extra parallel `data_num` trailing → `parallel = inst_count * data_num`
- `CircuitCore1T1R.cim_read`: parallel multiplicity is broadcast-determined per call (the B-side `b_positions` from `classify_leading_positions`). Equivalent formulation: `serial = prod(leading[p] for p in a_positions)` — only x-side A positions are serial; B-side instances are parallel physical xbars and do **not** enter the per-op latency count.

## Composite-module PPA aggregation

When a composite circuit (`CircuitCore1T1R`, `OffsetSwitchCapMuxAdcReadOut`, `Offset1T1RXbar`) owns child circuit modules:

- Composite's `area_per_inst__um2` / `leakage_per_inst__uW` return **only the composite's own extra silicon cost** (`self.config.*` directly).
- Each child contributes through its own `inst_area__um2` / `inst_leakage__uW` when `collect_static` walks the model graph.

Latency composition follows the same pattern but **at the latency event stream**:

- Composites that emit dynamic events (`CircuitCore1T1R`, `OffsetSwitchCapMuxAdcReadOut`) read **only the composite's own per-op overhead** from `self.config.latency_per_op__ns` (orch / glue logic), not the sum of children's latencies.
- When the composite's primary method runs, it calls `_log_dynamic_energy(dynamic_energy_self)` and / or `_log_latency(latency_self)` for its own overhead. Children **that have their own dynamic profile model** (e.g. a downstream DAC / ADC / digital block) call the same methods for their own contributions; children without a dynamic model (devices, dynamics-less analog blocks) contribute through the composite's own energy / latency tensors instead.
- `profiler.total_latency__ns` sums every `LatencyEvent` across the whole stream — yielding the actual pipeline contribution sum without any composite-side summation.

This is the central correctness invariant: composite circuits **do not** sum the children's emitted contributions into their own emission, because that would double-count what the children already log. The profiler-level event aggregation is the single source of truth.

## Devices and macros

Devices (`RRAM`, `NMOS`, `Selector`) do not emit profile events at all. Their physical contribution (e.g. RRAM read settling time inside the array) is already folded into the owning circuit's `CircuitCore1T1RConfig.latency_per_op__ns`. They have no separate per-op latency to log.

Macros (`XbarMacro` family) are orchestration nodes; they inherit `ProfileMixin` only for the name + module_type machinery so children can be named under them. They contribute zero PPA themselves; everything flows through their constituent circuits.

## `total_latency__ns` semantics

`profiler.total_latency__ns` (and the derived `ProfilerReport.total_latency__ns`) is the **sum of per-event latency contributions** — equivalent to assuming sequential execution of every recorded operation. It is **not** a wall-clock pipeline latency: a parallel / pipelined schedule would yield a smaller end-to-end time. The contribution sum is what feeds `ProfilerReport.leakage_energy__fJ = static.leakage_power__uW × total_latency__ns`.

## Profiler entry points

`NeuroxProfiler` is a strict context manager. Inside the `with` block, every `_log_dynamic_energy` / `_log_latency` call pushes a 0-D tensor into a per-quantity pending buffer; outside the block the calls are no-ops. **Reading aggregations or calling `report()` from inside the `with` block is unsupported** — pending events have not been synced yet. Read after the block exits, when `__exit__` has invoked `_finalize`.

Public methods:

- `energy_events: list[EnergyEvent]` / `latency_events: list[LatencyEvent]` — runtime events, populated by `_finalize`.
- `total_dynamic_energy__fJ` / `total_latency__ns` — pre-computed scalar sums.
- `energy_by_name` / `energy_by_type` / `latency_by_name` — pre-computed grouped sums.
- `collect_static(model) -> list[StaticRecord]` — walk `model.modules()` for every `CircuitBase` instance, read its `inst_area__um2 / inst_leakage__uW`.
- `analyze_static(model) -> StaticMetrics` — aggregate to total area + leakage; `StaticMetrics` carries only area + leakage.
- `report(model) -> ProfilerReport` — combine runtime events + static walk. `ProfilerReport.total_latency__ns` is a derived property over `latency_events`; `leakage_energy__fJ` reads it × `static.leakage_power__uW`.

`_finalize` is private — driven only by `__exit__` on clean context exit. One batched GPU→CPU sync per quantity. See [`modules/profiler/README.md`](../modules/profiler/README.md) for the pending-buffer + sync mechanics.

## See also

- [`docs/dev/modules/common/circuit.md`](../modules/common/circuit.md) — `CircuitBase` / `CircuitConfig` dev doc
- [`docs/dev/modules/profiler/README.md`](../modules/profiler/README.md) — profiler internals
- [`docs/dev/architecture/compile_policy.md`](compile_policy.md) — why the log methods are `@torch.compiler.disable`
- [`docs/dev/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md`](../adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md) — device vs. circuit split

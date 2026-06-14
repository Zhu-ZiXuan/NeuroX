# Profiler Modules

The per-module PPA / energy / latency side-channel lives in:

- `neurox/common/mixin/profile.py` — `ProfileMixin`, the small mixin that gives a module a hierarchical profiler name + `module_type` + two independent log entries: `_log_dynamic_energy(energy)` and `_log_latency(latency)`. Carries no PPA awareness; static-PPA fields live on `CircuitBase`.
- `neurox/common/circuit.py` — `CircuitBase[ConfigT]`, the composition every electrical circuit inherits. Provides the typed static-PPA surface (area + leakage) backed by `self.config` (a `CircuitConfig`).
- `neurox/common/profiler.py` — `NeuroxProfiler` context manager that batches the events and produces per-layer PPA reports.

Reach the public surfaces through `neurox.common.CircuitBase`, `neurox.common.CircuitConfig`, `neurox.common.mixin.ProfileMixin`, and `neurox.common.profiler.NeuroxProfiler`.

## Hierarchical naming

A profiled module receives a hierarchical name from its constructor (e.g. `<layer>.<owner>.<member>` — `fc1.foo.bar.baz`), and each child circuit appends its own role to that name when it constructs further children. Devices do **not** carry profiler names — their PPA rolls up to the owning circuit.

## Side-channel only

Every numerical return value in NeuroX is the actual analog / digital result; energy and latency travel through the profiler side channel only. That keeps the hot-path tensor pipeline narrow and `@torch.compile`-clean.

## Static vs dynamic

- **Static** PPA (area, leakage power) is the per-instance property surface on `CircuitBase`. The profiler walks `model.modules()`, sums `inst_area__um2` / `inst_leakage__uW` over every `CircuitBase` instance, and reports the total once per session. No per-call work.
- **Dynamic** energy and latency are independent per-call quantities: a leaf that has a dynamic profile model calls `_log_dynamic_energy(energy_tensor)` and / or `_log_latency(latency_tensor)` once at the end of its primary method. Either, both, or neither — the leaf decides based on its physical model; leaves without a dynamic model (devices, dynamics-less analog blocks) emit nothing.

## Two independent log entries

```python
class ProfileMixin:
    def _log_dynamic_energy(self, dynamic_energy__fJ: Tensor) -> None: ...
    def _log_latency(self, latency__ns: Tensor) -> None: ...
```

Both arguments are **tensors** (not Python floats); the log functions only route the tensor to the active profiler — they perform no value-based gating. Any "should I emit?" decision lives at the call site.

Fixed-latency leaf pattern (energy + latency both nonzero):

```python
def some_op(self, ...):
    y = ...
    op_count = self._serial_op_count(y)   # leaf-specific shape math
    dynamic_energy__fJ = torch.full_like(y, self.config.energy_per_op__fJ)
    latency__ns        = torch.tensor(self.config.latency_per_op__ns * op_count,
                                      device=y.device, dtype=dynamic_energy__fJ.dtype)
    self._log_dynamic_energy(dynamic_energy__fJ)
    self._log_latency(latency__ns)
```

Parametric-latency leaf (e.g. SAR ADC) only changes how `per_op_latency__ns` is computed — `(adc_operation_point.adc_bits + 1) × config.clk_period__ns` in place of `self.config.latency_per_op__ns`. Same two log calls.

Composite-orch leaf (e.g. readout) gates each quantity independently — a latency-only orch records its latency event even when `energy_per_op__fJ` is 0:

```python
if self.config.energy_per_op__fJ > 0.0:
    self._log_dynamic_energy(...)
if self.config.latency_per_op__ns > 0.0:
    self._log_latency(...)
```

Both quantities walk identical paths inside the profiler: `.detach().sum()` on entry (reduces to a 0-D tensor on the recording device), then one batched `stack → cpu → tolist` sync per quantity at `_finalize`. There is no per-call `.item()` and no per-event GPU sync — pipeline cost is independent of event count.

`EnergyEvent.dynamic_energy__fJ` and `LatencyEvent.latency__ns` are `float` for external consumers; the tensors only exist inside the recording buffer.

## Per-VMM aggregation (composite-forward modules)

Composite forward modules whose body contains an internal iteration / chunked loop (currently `CircuitCore1T1R.cim_read` is the only one) must emit exactly **one** energy event + **one** latency event per logical VMM, regardless of how many sub-solver calls or chunks are inside. The pattern is:

1. Aggregate per-chunk / per-iter energy inside the loop (`_compute_array_energy__fJ` per chunk).
2. After the loop, reassemble per-chunk energies to the full leading shape.
3. Build the latency tensor from `self.config.latency_per_op__ns × serial_op_count` where `serial_op_count = prod(leading[p] for p in a_positions)` — only the x-side A positions count as serial; B-side (inst) positions are parallel physical hardware and are excluded. Same convention as every other emitting leaf (which divides `output.numel()` by `inst_count`).
4. Call `_log_dynamic_energy(energy)` and `_log_latency(latency)` exactly once at the end of the forward body — same inline pattern as every other plain forward leaf. Sub-solvers and devices inside the loop do **not** emit profile events of their own (they are not `CircuitBase`), so there is no double-counting risk.

Non-composite leaves (DAC, ADC, accumulator, mux, switch-cap, …) call the two log entries inline at the end of forward — single call, no loop.

`tests/test_xbar_chunking.py::test_profiler_single_event_under_chunking` verifies the contract end-to-end (one energy event + one latency event per VMM, regardless of chunking).

## Usage contract: `with` block first, then read

`NeuroxProfiler` is a strict context manager:

```python
with NeuroxProfiler() as profiler:
    model(...)
# __exit__ runs _finalize() — exactly one batched GPU→CPU sync per quantity.
print(profiler.total_dynamic_energy__fJ, profiler.total_latency__ns)
report = profiler.report(model)
```

Reading any aggregation property or calling `report()` **inside** the `with` block is unsupported — pending events are still on the recording device, so the values are stale / incomplete. Always finish recording (exit the block) before reading.

`__exit__` calls `_finalize()` automatically on clean exit (skipped on exception). After that, every property is a plain field access — zero GPU work.

`ProfilerReport.total_latency__ns` is derived from `latency_events` on demand; `StaticMetrics` carries only area + leakage. `ProfilerReport.leakage_energy__fJ = static.leakage_power__uW × total_latency__ns` is the single derivation point.

## `collect_static` walks `CircuitBase`

```python
@staticmethod
def collect_static(model: nn.Module) -> list[StaticRecord]:
    return [
        StaticRecord(
            qualified_name=m.qualified_name,
            module_type=m.module_type,
            area__um2=m.inst_area__um2,
            leakage_power__uW=m.inst_leakage__uW,
        )
        for m in model.modules()
        if isinstance(m, CircuitBase)
    ]
```

The judgement is `isinstance(m, CircuitBase)` — `CircuitBase` is the layer that carries the typed static-PPA surface. Pure `ProfileMixin` subclasses (e.g. macros, which only need name + dynamic events) don't appear in the static report because they have no static PPA. Devices (`RRAM` / `NMOS` / `Selector`) don't appear either: their physical contribution is already folded into the owning circuit's `CircuitConfig`.

`inst_area__um2` and `inst_leakage__uW` are derived on access from `area_per_inst__um2 × inst_count` (no cache).

## See also

- [`docs/dev/modules/common/circuit.md`](../common/circuit.md) — `CircuitBase` / `CircuitConfig` dev doc
- [`docs/dev/architecture/profiler_and_ppa.md`](../../architecture/profiler_and_ppa.md) — full PPA + event rules across the layer hierarchy
- [`docs/dev/architecture/chunking.md`](../../architecture/chunking.md) — chunking strategy
- [`docs/dev/roadmap.md`](../../roadmap.md) — CSV / JSON export and benchmark suite plans

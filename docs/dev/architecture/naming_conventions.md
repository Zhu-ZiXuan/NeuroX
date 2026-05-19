# Naming Conventions

This document collects the project-wide naming rules. Every new symbol must follow them.

## Physical-quantity names

Variables and fields that carry a physical quantity use the `<name>__<unit>` form described in [`physical_units.md`](physical_units.md). Dimensionless names take no suffix.

## Class suffixes by role

Dataclasses returned or accepted at module boundaries use one of the following semantic suffixes. The suffix tells the reader what the type *is*; the body explains what it *carries*.

| Suffix | Meaning | Examples |
|---|---|---|
| `*Config` | Frozen design / spec configuration for a circuit, device, or family. | `NMOSConfig`, `OpAmpTIAConfig`, `OffsetSwitchCapMuxAdcReadOutConfig` |
| `*Snapshot` | Per-call runtime snapshot of a module's working state, sampled at `snapshot(*, shape=...)` time. Carries only `Tensor` fields and nested `*Snapshot` instances. Frozen. | `NMOSSnapshot`, `RRAMSnapshot`, `OpAmpTIASnapshot`, `DriverSnapshot` |
| `*DCOP` | DC operating point — return type of any `solve_dc(...)` method. Carries the solved electrical quantities (voltages, currents, sensitivities). Frozen. | `NMOSDCOP`, `RRAMDCOP`, `OpAmpTIADCOP`, `DriverDCOP`, `Core1T1RDCOP`, `Solver1T1RDCOP` |
| `*Plan` | Static geometry / decomposition plan computed once and reused per execution. Frozen. | `TilePlan`, `SlicingPlan` |
| `*Result` | Result of an offline algorithm or iterative solver loop (i.e. neither runtime snapshot, nor DC operating point, nor a static plan). Frozen. | `CalibrationResult` |

When a return type is just one tensor, return the tensor directly — do not wrap it in a one-field dataclass.

## Buffer / attribute names

Three lifecycle stages exist for a module's physical state ([`state_holding.md`](state_holding.md)). Each stage uses a different name form:

| Stage | Form | When written |
|---|---|---|
| Nominal value (design intent) | `nominal_<name>__<unit>` for Tensor buffers; plain `<name>__<unit>` for Python scalars | `__init__` for circuits; first step of `program(...)` for RRAM-style devices |
| Actual value (post-fabrication, with static mismatch) | `<name>__<unit>` (no prefix) | `fabricate(...)` / `program(...)` |
| Snapshot value (with dynamic noise) | Fields of the `*Snapshot` dataclass returned by `snapshot(*, shape=...)` | `snapshot(...)`; never registered as a buffer |

**SwitchCap exception**: `c_unit__fF` is the unit capacitance value (a PDK constant), not a design-stage nominal. It does not take the `nominal_` prefix. Modules whose shape and topology are themselves runtime fabrication arguments may legitimately have no stable nominal buffer.

## Primary-method names

Each circuit / device class exposes one primary method whose name encodes the physical operation it performs. These names are reserved:

| Method | Semantics |
|---|---|
| `fabricate(shape, **extras) -> None` | Sample static per-instance state over the given instance shape. Re-callable. Must end with `self._record_inst_count(shape)` for circuit modules (see [`profiler_and_ppa.md`](profiler_and_ppa.md)). |
| `program(...) -> None` | RRAM-specific weight programming step that takes the integer weight tensor and produces the actual conductance buffer. |
| `snapshot(*, shape) -> <Name>Snapshot` | Sample a per-call runtime snapshot. Frozen return. |
| `solve_dc(...) -> <Name>DCOP` | Solve the DC operating point of a circuit or array. Naming is uniform across leaf devices, leaf circuits, composite circuits, and solver classes. |
| `solve_clamp(...) -> tuple[Tensor, Tensor]` | `ClampDriver`-protocol entry. Returns `(v_clamp__V, dVclamp_dI__MOhm)` for use by outer solvers. |
| `convert(...) -> Tensor` | DAC / ADC code↔analog conversion. Single output tensor. |
| `transport(...)` | AnalogMux differential voltage transport. |
| `drive(...)` | Decoder WL drive entry point. |
| `sample_and_accumulate(...) -> Tensor` | SwitchCap passive charge-share kernel. |
| `vec_mat_mul(x) -> Tensor` | Xbar tile per-VMM kernel. |
| `readout(...) -> Tensor` | Readout chain entry. Returns only the ADC code tensor. |
| `operate(...) -> Tensor` | Digital block primary kernel (adder / accumulator / requantizer / shift-adder / subtractor). |
| `matmul(...) -> Tensor` | Macro entry point. The single `@torch.compile` boundary of the project (see [`compile_policy.md`](compile_policy.md)). |

## Module hierarchy names (profiler)

Every `ProfiledModule` receives a hierarchical `name` constructed by the owning parent. Names use `.` as separator and reflect the ownership tree:

```
<layer>.<owner>.<leaf>
fc1.macro.xbar.core.tia
```

Children always derive their name from the parent's name plus a fixed local suffix. Local suffixes mirror attribute names on the parent.

## Config layout

Inside each `*Config`, fields are grouped by `validate_<group>` methods. Field order in source follows the same grouping. See [`config_and_construction.md`](config_and_construction.md).

## Module file naming

Source files are `lower_snake_case.py`. Class names inside use `UpperCamelCase`. There is no `*Impl` / `*Base` / `*Abstract` suffix on classes — abstract bases use the bare family name (`ADC`, `DAC`, `TIA`, `Xbar`, `ReadOut`); concrete impls use a descriptive name (`McsSarAdc`, `GeneralDAC`, `OpAmpTIA`, `Offset1T1RXbar`, `OffsetSwitchCapMuxAdcReadOut`).

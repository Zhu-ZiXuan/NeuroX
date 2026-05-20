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
| `*Plan` | Static geometry / decomposition plan computed once and reused per execution. Frozen. | `SlicingPlan` |
| `*Result` | Result of an offline algorithm or iterative solver loop (i.e. neither runtime snapshot, nor DC operating point, nor a static plan). Frozen. | `CalibrationResult` |

When a return type is just one tensor, return the tensor directly — do not wrap it in a one-field dataclass.

## Buffer / attribute names

Three lifecycle stages exist for a module's physical state ([`state_holding.md`](state_holding.md)). Each stage uses a different name form:

| Stage | Form | When written |
|---|---|---|
| Nominal value (design intent) | `nominal_<name>__<unit>` for Tensor buffers; plain `<name>__<unit>` for Python scalars | `__init__` for circuits; first step of `program(...)` for RRAM-style devices |
| Actual value (post-fabrication, with static mismatch) | `<name>__<unit>` (no prefix) | `fabricate(...)` / `program(...)` |
| Snapshot value (with dynamic noise) | Fields of the `*Snapshot` dataclass returned by `snapshot(*, shape=...)` | `snapshot(...)`; never registered as a buffer |

PDK-constant scalars surfaced as buffers (e.g. SwitchCap's `cfg.c_unit__fF` accessed directly through `cfg`) do not need a `nominal_` prefix — the nominal-vs-actual distinction applies to fabricated arrays, not to a single PDK datum that never gets perturbed.

## Primary-method names

Each circuit / device class exposes one primary method whose name encodes the physical operation it performs. These names are reserved:

| Method | Semantics |
|---|---|
| `fabricate(shape) -> None` | Sample static per-instance state over the given instance shape. Re-callable. Must end with `self._record_inst_count(shape)` for circuit modules (see [`profiler_and_ppa.md`](profiler_and_ppa.md)). Structural facts threaded through `__init__` instead — `fabricate` is shape-only. |
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

Source files are `lower_snake_case.py`. Class names inside use `UpperCamelCase`. There is no `*Impl` / `*Base` / `*Abstract` suffix on classes — abstract bases use the bare family name; concrete implementations use a descriptive name without a suffix that marks them as concrete.

## Import paths and subpackage exports

A class defined inside a subpackage must be imported from that subpackage's path, never re-exported by an ancestor package's `__init__.py`. The directory layout is the import path.

Given a package layout `pkg/sub/<file>.py` declaring class `<Class>`:

- The required import is `from pkg.sub import <Class>`. The subpackage's own `__init__.py` may re-export classes defined in files at its own directory level.
- `from pkg import <Class>` is disallowed. The parent `pkg/__init__.py` must not re-export anything reached through a subdirectory.

Each package's `__init__.py` re-exports only the symbols defined in files at its own directory level — never anything reached through a subdirectory.

Rationale: the import path mirrors the source layout, so a reader can find any symbol by walking the directory tree, and an `__init__.py` never accumulates an unbounded re-export list as new subpackages get added.

### Exceptions

Two narrow exceptions allow a parent package to re-export from a subdirectory. Both are deliberate API choices, not generic permission.

1. **Subpackages whose directory name cannot be a Python identifier.** When a subdirectory must carry a leading underscore only because the natural name would start with a digit, that underscored name is a Python-level workaround that should not leak into the public API. The parent package re-exports the subpackage's small public surface so callers do not have to spell the workaround.

2. **Subpackages that exist solely to organise internal implementation files.** When a subpackage groups multiple impls plus shared private helpers, and the impls themselves form the parent package's intended public surface, the parent re-exports those impls and hides the subpackage path. Callers see one flat public surface; the subdirectory remains an organisational tool.

In both cases:

- Only the user-facing classes are re-exported, never internal helpers / solvers / shared utilities.
- The exception applies to a fixed list of subpackages, declared in this document. New subpackages do not get the exception by default — the regular rule applies unless a new entry is added here.

Current exceptions: `neurox.xbar` (reason 1) and `neurox.operator` (reason 2).

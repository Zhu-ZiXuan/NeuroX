# Naming Conventions

This document collects the project-wide naming rules. Every new symbol must follow them. These are code-level conventions; the equation-side notation lives in [reference/notation_conventions](../reference/notation_conventions.md).

## Physical-quantity names

Variables and fields that carry a physical quantity use the `<name>__<unit>` form. Unit suffixes are ASCII, matching the documented set: `uA`, `uS`, `um`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`. Dimensionless names take no suffix. The equation-side units and symbols for these same quantities live in [reference/notation_conventions](../reference/notation_conventions.md).

### Suffix grammar

The `<name>__<unit>` suffix follows a fixed grammar:

- Separator: double underscore `__` between name and unit.
- Unit-symbol case follows the physical standard (`V` uppercase for volt, `m` lowercase for metre; SI scale prefixes `T G M k m u n p f` keep their standard case).
- Multiplication is implicit — two adjacent unit symbols denote a product, e.g. `A_vt__mV_um` is mV × um.
- Division uses the explicit `_per_` token, e.g. `mu0__cm2_per_V_s` is cm^2 / V / s.
- Config-surface fields may use industrial units outside the runtime set (e.g. `mV`, `cm2`); the runtime tensor set is the closed list above.

| Name | Reading |
|---|---|
| `v_dd__V` | supply voltage in volts |
| `T_ref__K` | reference temperature in kelvin |
| `c_unit__fF` | unit capacitance in femtofarads |
| `A_vt__mV_um` | Pelgrom V_th coefficient in mV × um |
| `mu0__cm2_per_V_s` | low-field mobility in cm^2 / V / s |
| `K_BOLTZMANN__J_per_K` | Boltzmann constant in J / K |

## Class suffixes by role

Dataclasses returned or accepted at module boundaries use one of the following semantic suffixes. The suffix tells the reader what the type *is*; the body explains what it *carries*.

| Suffix | Meaning | Examples |
|---|---|---|
| `*Config` | Frozen design / spec configuration for a circuit, device, or family. | `NMOSConfig`, `OpAmpTIAConfig`, `OffsetSwitchCapMuxAdcReadOutConfig` |
| `*Snap` | Per-call runtime snap of a module's working state, sampled at `snapshot(*, shape=...)` time. Carries only `Tensor` fields and nested `*Snap` instances. Frozen. | `NMOSSnap`, `RRAMSnap`, `OpAmpTIASnap`, `VoltageDriverSnap` |
| `*DCOP` | DC operating point - return type of any `solve_dc(...)` method. Carries the solved electrical quantities (voltages, currents, sensitivities). Frozen. Used only when the module truly owns a DC operating point (devices, dedicated solvers, the TIA's op-amp clamp). Composite forward modules (e.g. `Core1T1R`) do **not** define one; they return primary output tensors (or a plain steady-state struct) and log energy/latency inline. | `NMOSDCOP`, `RRAMDCOP`, `OpAmpTIADCOP`, `SolverDCOP` |
| `*Plan` | Static geometry / decomposition plan computed once and reused per execution. Frozen. | - |
| `*Result` | Result of an offline algorithm or iterative solver loop (i.e. neither runtime snap, nor DC operating point, nor a static plan). Frozen. | `CalibrationResult` |

When a return type is just one tensor, return the tensor directly - do not wrap it in a one-field dataclass.

`snapshot` is reserved for the verb (the act of sampling) and the producing method `snapshot(*, shape=...)`. Every noun — the per-call state, the `*Snap` dataclass and its instances, the lifecycle stage, the concept — is a `snap` in prose, comments, and identifiers (`cell_snap`, `nmos_snap`, the `snap` parameter of `solve_dc` / `solve_clamp`). The method name never shortens to `snap`.

## Buffer / attribute names

Three lifecycle stages exist for a module's physical state. Each stage uses a different name form:

| Stage | Form | When written |
|---|---|---|
| Nominal value (design intent) | `nominal_<name>__<unit>` for Tensor buffers; plain `<name>__<unit>` for Python scalars | `__init__` for circuits; first step of `program(...)` for RRAM-style devices |
| Actual value (post-fabrication, with static mismatch) | `<name>__<unit>` (no prefix) | `fabricate(...)` / `program(...)` |
| Snap value (with dynamic noise) | Fields of the `*Snap` dataclass returned by `snapshot(*, shape=...)` | `snapshot(...)`; never registered as a buffer |

PDK-constant scalars surfaced as buffers (e.g. SwitchCap's `config.c_unit__fF` accessed directly through `config`) do not need a `nominal_` prefix - the nominal-vs-actual distinction applies to fabricated arrays, not to a single PDK datum that never gets perturbed.

## Primary-method names

Each circuit / device class exposes one primary method whose name encodes the physical operation it performs. These names are reserved:

| Method | Semantics |
|---|---|
| `fabricate() -> None` | Inherited from `FabricateMixin`; auto-cascades the static-mismatch resample across self + children. Subclasses override `_sample_fabricate_mismatch(self)` only. Per-instance shape is bound at `__init__` via `inst_shape` (leaves and xbars) or `w_logical_shape` (macros). Static PPA is exposed as `CircuitBase` properties reading `self.config` - no per-init log call. |
| `program(...) -> None` | RRAM-specific weight programming step that takes the integer weight tensor and produces the actual conductance buffer. |
| `snapshot(*, shape) -> <Name>Snap` | Sample a per-call runtime snap. Frozen return. |
| `solve_dc(...) -> <Name>DCOP` | Solve the DC operating point of a circuit, device, or solver and return it as a `*DCOP`. The essence is "compute a meaningful DC operating point and surface it" - applicable to leaf devices (RRAM, NMOS), iterative dedicated solvers (`NestedParallelRailSolver`, `OpAmpTIA`), and the boundary clamp drivers (`TIA`, `VoltageDriver`). **Composite forward modules that delegate to sub-solvers and add post-processing do not own a DCOP and do not use this name** - see `solve_array` below. |
| `solve_clamp(...) -> tuple[Tensor, Tensor]` | Boundary-clamp solve. Thin wrapper around the implementer's DC solve, returning the `(v_clamp__V, dVclamp_dI__MOhm)` pair an outer solver needs as Jacobian input. Both `TIA` and `VoltageDriver` expose this. |
| `solve_array(v_wl, *, bl_driver, sl_driver, ...) -> CoreSteadyState` | `Core1T1R` entry. Take the analog WL drive and the two boundary clamp drivers, settle the 1T1R array to DC, and return a plain `CoreSteadyState` (BL port current + BL clamp voltage). Plain forward (logs energy + latency inline); does not return a DCOP - the array has no DCOP of its own beyond its sub-solvers' DCOPs. |
| `convert(...) -> Tensor` | DAC / ADC code/analog conversion. Single output tensor. |
| `transport(...)` | VoltageMux differential voltage transport; reused by `CurrentMux` for single-ended N:1 current transport. |
| `replicate(...) -> Tensor` | `CurrentMirror` current-copy replication. |
| `sample_and_accumulate(...) -> Tensor` | SwitchCap passive charge-share kernel. |
| `vec_mat_mul(x) -> Tensor` | Xbar tile per-VMM kernel. |
| `readout(...) -> Tensor` | Readout chain entry. Returns only the ADC code tensor. |
| `operate(...) -> Tensor` | Digital block primary kernel (adder / accumulator / shift-adder / subtractor). |
| `matmul(...) -> Tensor` | Macro entry point. Eager (compile-friendly); the library self-compiles only the solver leaf. |

## Module hierarchy names (profiler)

Every `ProfileMixin` receives a hierarchical `name` constructed by the owning parent. Names use `.` as separator and reflect the ownership tree:

```text
<layer>.<owner>.<leaf>
fc1.macro.xbar.core.tia
```

Children always derive their name from the parent's name plus a fixed local suffix. Local suffixes mirror attribute names on the parent.

## Config layout

Inside each `*Config`, fields are grouped by `validate_<group>` methods. Field order in source follows the same grouping.

## Module file naming

Source files are `lower_snake_case.py`. Class names inside use `UpperCamelCase`. There is no `*Impl` / `*Base` / `*Abstract` suffix on classes - abstract bases use the bare family name; concrete implementations use a descriptive name without a suffix that marks them as concrete.

## Import paths and subpackage exports

A class defined inside a subpackage must be imported from that subpackage's path, never re-exported by an ancestor package's `__init__.py`. The directory layout is the import path.

Given a package layout `pkg/sub/<file>.py` declaring class `<Class>`:

- The required import is `from pkg.sub import <Class>`. The subpackage's own `__init__.py` may re-export classes defined in files at its own directory level.
- `from pkg import <Class>` is disallowed. The parent `pkg/__init__.py` must not re-export anything reached through a subdirectory.
- Import from the subpackage, never drill into its file: `from neurox.xbar._1t1r import XbarCell1T1R`, never `from neurox.xbar._1t1r.cell import XbarCell1T1R`. The file is an implementation detail; the subpackage `__init__` is the surface.

Each package's `__init__.py` re-exports only the symbols defined in files at its own directory level - never anything reached through a subdirectory.

Rationale: the import path mirrors the source layout, so a reader can find any symbol by walking the directory tree, and an `__init__.py` never accumulates an unbounded re-export list as new subpackages get added.

Test modules (under `tests/`) and tool / CLI modules (under `neurox/tools/`) are exempt from this file-vs-subpackage rule: they may import directly from a file path (for example `from neurox.xbar._1t1r.cell import XbarCell1T1R`), since they are not part of the library public surface and sometimes need a symbol the subpackage `__init__` does not re-export.

### Exceptions

Two narrow exceptions allow a parent package to re-export from a subdirectory. Both are deliberate API choices, not generic permission.

1. **Subpackages whose directory name cannot be a Python identifier.** When a subdirectory must carry a leading underscore only because the natural name would start with a digit, that underscored name is a Python-level workaround that should not leak into the public API. The parent package re-exports the subpackage's small public surface so callers do not have to spell the workaround.

2. **Subpackages that exist solely to organise internal implementation files.** When a subpackage groups multiple impls plus shared private helpers, and the impls themselves form the parent package's intended public surface, the parent re-exports those impls and hides the subpackage path. Callers see one flat public surface; the subdirectory remains an organisational tool.

In both cases:

- Only the user-facing classes are re-exported, never internal helpers / solvers / shared utilities.
- The exception applies to a fixed list of subpackages, declared in this document. New subpackages do not get the exception by default - the regular rule applies unless a new entry is added here.

Current exceptions: `neurox.xbar` (reason 1).

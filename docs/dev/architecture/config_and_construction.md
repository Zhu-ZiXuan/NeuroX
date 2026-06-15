# Config and Construction

This document records the current construction rules for NeuroX's device and analog / circuit hierarchy.

`config` and `policy` are the two parameter objects every module accepts at `__init__`:

- `config` (a `*Config` dataclass) describes **what the module is**: physical parameters, design parameters, spec / PPA values. Static across deployments; persists in preset TOML.
- `policy` (a `*Policy` dataclass) describes **how the module should behave at runtime**: which non-idealities to apply, and (future) other behavioural switches. Per-run choice; constructed by the caller in code; never persisted in TOML.

The two are strongly coupled — every module that has a `*Policy` also has a `*Config`, and policy fields gate parameters declared on the config — so they are described together in each module's doc.

## Parameter classes (config side)

NeuroX distinguishes four parameter classes on the `config` side:

1. **Process parameters** Fixed by process selection. They should change only when the process changes.
2. **Design parameters** Freely chosen within one process. Examples: transistor `W/L`, capacitor sizing, reference voltages, op-amp target gain.
3. **Spec parameters** Performance / statistics / bookkeeping parameters. Examples: noise, mismatch, area, leakage, latency.
4. **Runtime parameters** Only known when code runs. Examples: shape, active ADC mode, active bit width, operating temperature if treated as environment, and per-call tensors.

Runtime parameters never belong in config dataclasses.

## Device layer

Device configs contain only:

- process parameters
- spec parameters

Device design parameters are explicit `__init__` arguments.

Example:

- `NMOSConfig` contains process + mismatch parameters.
- `NMOS.__init__(..., W__um, L__um)` receives design parameters.

The reason is simple: many device instances share one process config while using different design sizes.

## Circuit layer

Circuit configs contain:

- design parameters
- spec parameters
- member configs for owned devices / child circuits

Circuits own their members and construct them directly. External factory closures are not part of the current design.

If a circuit contains a device, then that device's design parameters belong to the circuit config, not to the device config.

Example:

- `OpAmpTIAConfig` contains `nmos_config: NMOSConfig`.
- `OpAmpTIAConfig` also contains the pseudo-NMOS sizing parameters because they are part of the TIA design.
- `OpAmpTIA.__init__` constructs its internal `NMOS` directly from the config.

## Config tree mirrors ownership tree

The config tree should match the object ownership tree.

If object `A` owns `B`, then `A`'s config should usually contain `B`'s config as a field. The parent object then constructs `B` itself.

This keeps the design visible in one place:

- ownership is readable from the config tree
- construction is readable from the owning class
- no hidden factory closure decides what child class gets instantiated

## Per-instance shape at construction

Every fabricable module takes its per-instance fabrication shape as a constructor argument, not a `fabricate(...)` argument. The argument name varies by layer:

- leaf circuits (analog / digital / device): `inst_shape: tuple[int, ...]`
- xbar tiles: `inst_shape: tuple[int, ...]` (per-instance multiplicity prefix; trailing `(col_num, w_digit_count, row_num)` is owned by the xbar's own config)
- xbar macros: `w_logical_shape: tuple[int, ...]` (operator-facing weight shape)

The shape is committed once at `__init__`, recorded on `self._inst_shape` (per the `FabricateMixin` contract; circuits inherit this hook through `CircuitBase`). `fabricate()` then carries no arguments; it resamples mismatch at the already-bound shape. See [`fabrication_lifecycle.md`](fabrication_lifecycle.md) for the lifecycle. The profiler computes the instance count on demand from `_inst_shape` — see [`profiler_and_ppa.md`](profiler_and_ppa.md) and [`modules/common/circuit.md`](../modules/common/circuit.md).

## Family bases, concrete configs, and dispatch

Families with multiple concrete implementations use:

- one abstract base class
- one base config class
- one concrete config class per concrete implementation
- one concrete implementation class per concrete config

Each concrete implementation registers its config type on the family base via `RegistryMixin[type[<Family>Config], <Family>]`.

The base class exposes a family-specific `from_config(...)` classmethod that materialises the registry key from a config instance (`type(config)`) and instantiates the impl. `RegistryMixin` does not provide `from_config(...)`; it only provides:

- `register_key(...)`
- `_lookup_impl(...)`

This keeps dispatch generic while preserving explicit family-level runtime arguments.

## Explicit `from_config`

`from_config(...)` exists on family bases rather than on config classes.

Why:

- the family base owns the mapping from config type to implementation type
- a parent object usually knows it wants "some `ADC`" or "some `TIA`"
- a config object should not instantiate arbitrary unrelated families

`from_config(...)` must use explicit parameters only. No hidden defaults. This prevents silent divergence between families and avoids partially-applied construction rules.

Different families may expose different explicit runtime parameters.

Example:

- `XbarMacro.from_config(...)` carries `ideal_xbar`
- `TIA.from_config(...)` does not

The project does **not** force one global `from_config(...)` signature across all families.

## Mixin rule

`RegistryMixin[KeyT, ImplT]` is generic and only solves the registry / lookup problem. For config-class-keyed dispatch the family parametrises it as `RegistryMixin[type[<Family>Config], <Family>]`; for string-discriminator dispatch (e.g. Transcoder) the family parametrises it as `RegistryMixin[<DiscriminatorLiteral>, <Family>]`.

Each family declares its own:

- factory classmethod (`from_config(...)`, `create(...)`, etc.)
- family-specific constructor rules

The registry attribute `_impl_registry` is materialised automatically by the mixin per family root; families do not declare it by hand.

## Config validation

Every frozen config inherits `ValidateMixin` (from `neurox/common/mixin/validate.py`), either directly or transitively via `CircuitConfig` (see [`modules/common/circuit.md`](../modules/common/circuit.md) — `CircuitConfig` is the standard base for electrical-circuit configs and carries the area / leakage / latency fields). Validation runs at construction time via `__post_init__ → validate()`. Whether `__post_init__` is defined on the leaf or inherited from a circuit-family base is a **simplicity decision** for each family.

```python
from neurox.common.mixin import ValidateMixin


@dataclass(frozen=True)
class FooConfig(ValidateMixin):
    field_a: float
    field_b: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_group_a()
        self.validate_group_b()

    def validate_group_a(self) -> None:
        self._require_pos(self.field_a, "field_a")

    def validate_group_b(self) -> None: ...
```

Rules:

- Every frozen config **must inherit `ValidateMixin`** (transitively via a base config is fine). The mixin contributes only the runtime-check helpers; it does **not** define `__post_init__` or `validate` — keeping the mixin pure means every project-wide rule about *when* validation runs lives in this document, not in mixin source.
- Every config **must have a `validate()` method on the inheritance chain** — empty bodies are fine for marker bases, but the method must exist so subclasses can chain via `super().validate()`.
- Every config **must be validated at construction time** via a `__post_init__` that calls `self.validate()` and does nothing else. The entry point may live on the leaf or be inherited from a family base; choose by code simplicity.
- For circuit-family bases (`ADCConfig`, `TIAConfig`, `XbarConfig`, `XbarMacroConfig`, `Solver1T1RConfig`, `ReadOutConfig`, …), the family base owns `__post_init__` so every leaf inherits validation automatically and only needs to override `validate()` when it adds new checks. A leaf may still define its own `__post_init__` if that is locally clearer.
- Multi-field configs split `validate()` into one `validate_<group>()` per logical group of fields. Group names follow field semantics (`validate_process` / `validate_temperature` / `validate_mismatch` / `validate_ppa` / `validate_noise` / `validate_topology` / `validate_geometry` / …). Use the standard name `validate_ppa` for the area / leakage / latency trio.
- Single-field configs may put the check directly inside `validate()` without a group method.
- Subclasses' `validate()` must start with `super().validate()` before calling their own group methods. This is the only way to chain validation up the inheritance tree.
- A leaf with no new validation does **not** need to override `validate()` — the chain already runs.

### What `validate` is for

`validate*` methods encode **runtime checks that static analysis cannot express**:

- Numeric bounds (`> 0`, `>= 0`, `0 <= x <= 1`).
- Monotonicity (strictly increasing / decreasing sequences).
- Length constraints (`len(seq) >= N`).
- Cross-field relationships (`v_dd > v_ref`, `col_num % ref_group_size == 0`).

### What `validate` is NOT for

Static-typing-checked properties are NOT re-checked at runtime:

- `isinstance(x, float)` / `isinstance(x, int)` — type annotation handles it.
- `value in <Literal allowed set>` — enforced at TOML / YAML load time (see below) and statically inside Python-direct paths.
- Field shape / dtype on `Tensor`-typed fields (no such fields appear on frozen configs; tensors live on the consuming class as buffers).

If a check is already enforced by static analysis or by the deserialiser, do not duplicate it inside `validate*`.

### Literal at deserialization

`neurox/common/load_dump.py:_build_value` enforces `Literal[...]` allowed sets at TOML / YAML load time. A field annotated `Literal["mode_a", "mode_b"]` rejects unknown values during deserialisation; `validate*` does not need to repeat the check.

### Helpers

`ValidateMixin` (in `neurox/common/mixin/validate.py`) provides the runtime-check helpers as `@staticmethod` methods accessed via `self.`:

- `self._require_pos(value, name)`
- `self._require_nonneg(value, name)`
- `self._require_nonneg_or_none(value, name)`
- `self._require_pos_or_none(value, name)`
- `self._require_strictly_increasing(seq, name)`
- `self._require_strictly_decreasing(seq, name)`
- `self._require_min_length(seq, min_len, name)`

The mixin **does not** ship a `validate_ppa()` helper — the standard PPA trio is just three `self._require_nonneg(...)` calls inlined inside each config's own `validate_ppa()` group method.

Probability-distribution configs (`StuckAtFaultConfig`, `TelegraphConfig`, `LognormalConfig`, `GammaConfig`, `StateDependentGaussianConfig`, `StateDependentLognormalConfig`, `StateDependentGammaConfig`) live in `neurox/common/nonideality.py`; each inherits `ValidateMixin` and validates itself. Configs holding these distribution configs as fields **do not** re-validate them — the inner config's `__post_init__` already runs on construction.

## Module Policy

Every module that models non-idealities (or, in the future, any other runtime behavioural switch) carries a paired `*Policy` dataclass alongside its `*Config`. The policy is passed as the `policy=` kwarg at construction time.

### Rule

1. `*Config` holds **parameter values only** as fully-populated, non-Optional fields. Scalar sigmas are required `float`; multi-parameter distributions are required sub-config dataclasses. `None` is forbidden.
2. The decision of whether to apply a non-ideality (or other runtime switch) is carried by a separate **`*Policy`** dataclass passed as a kwarg to the owning module's `__init__`. Each module declares a `<Module>Policy` with one `bool` field per source.
3. The policy is **not** a config field, not a preset TOML entry, and not loaded from disk. It is a pure runtime parameter constructed by the caller at module instantiation time.
4. `*Policy` dataclasses have **no defaults** and **no factory methods** (no `all_off()` / `all_on()`). The caller must enumerate every field explicitly so that adding a new switch breaks every call site that has not yet declared a stance.
5. Runtime helpers in `neurox/common/nonideality.py` take a `*, enabled: bool` kw-only parameter and short-circuit to a pass-through when `enabled=False`. Callers therefore write a single unbranched expression — no `if`-gates at the call site.

### Why this split

The config conflated two distinct facts when an `enable_*` field lived on it: the *physical reality* of a noise source's magnitude (PDK datum) and the *study choice* of whether to model that source in this run. Splitting them isolates the concerns:

- `config` describes physical reality only. Process presets are reusable across deployments and studies.
- `Policy` describes the study choice. Different runs (calibration, training, inference, ablation) can pass different policies against the same config.
- Calibration tools construct an all-False policy directly, with no `dataclasses.replace(config, enable_*=False, ...)` indirection.
- The boundary between physical model and experimental decision is self-documenting in the constructor signature: `RRAM(config=..., policy=..., ...)`.

### Hierarchy: flat at leaves, structured at composites

Leaf modules have flat policies (a small dataclass of `bool` fields):

```python
@dataclass(frozen=True)
class RRAMPolicy:
    prog_gamma: bool
    stuck_at: bool
    read_telegraph: bool
    read_thermal: bool
```

Composite modules (e.g. `Xbar`, `XbarMacro`, `ReadOut`, `OpAmpTIA`, `CircuitCore1T1R`) hold structured policies that nest the leaf policies of their sub-modules, mirroring the config composition tree exactly:

```python
@dataclass(frozen=True)
class CircuitCore1T1RPolicy:
    rram: RRAMPolicy
    nmos: NMOSPolicy
    tia: TIAPolicy           # abstract; concrete impl passed
    sl_driver: DriverPolicy
    wl_dac: DACPolicy        # abstract; concrete impl passed
```

For polymorphic families (`ADC`, `DAC`, `TIA`, `ReadOut`, `Xbar`, `XbarMacro`) there is an empty abstract marker base policy and a concrete policy per registered impl. The composite that holds the polymorphic family stores the abstract base type and the caller passes the concrete impl that matches the config.

Modules with no behavioural switches (e.g. `IdealXbar`, `IdealXbarMacro`) declare an empty marker policy (`IdealXbarPolicy()` / `IdealXbarMacroPolicy()`) for API uniformity.

### Module attribute and naming

Each module stores its policy as `self.policy` (public attribute) so it is inspectable for debugging and profiling.

- Parameter field on config: `<source>__<unit>` when a physical unit exists, otherwise `<source>` (relative / probability / shape names).
- Policy field: `<source>`. Drop the `enable_` prefix — the policy class's role already implies "enable".

When a category sums multiple parameter fields, the policy field reflects the *category*, not any one parameter. Example: kT/C sampling noise in MCS-SAR and SwitchCap is gated by a single `sampling_thermal_noise: bool` even though the sigma is derived from per-instance capacitance.

### Runtime semantics

`apply_*` helpers in `nonideality.py` follow a uniform shape:

```python
def apply_<noise>(x: Tensor, ...args..., *, enabled: bool) -> Tensor:
    if not enabled:
        return x
    ...
```

Callers therefore write a single line per source:

```python
g__uS = apply_state_dependent_gamma(g__uS, self.config.prog_gamma, enabled=self.policy.prog_gamma)
g__uS = apply_telegraph_noise(g__uS, self.config.read_telegraph, enabled=self.policy.read_telegraph)
g__uS = apply_gaussian(g__uS, self.config.read_thermal__uS, enabled=self.policy.read_thermal)
```

No `if` branches at the call site, no `None` checks inside the helpers, no special-case dispatch.

Static-mismatch `apply_*` calls live inside `_sample_fabricate_mismatch()` (the `FabricateMixin` override point); dynamic per-call noise lives inside `convert` / `snapshot` / similar runtime methods. The policy fields are read directly there; the cadence at which the surrounding `fabricate()` is invoked is the operator-level concern documented in [`fabrication_lifecycle.md`](fabrication_lifecycle.md).

### What this rule does *not* apply to

- Numerical hyperparameters that are not noise (e.g. softclip softness, learning rates). These follow the regular "required field" rule but do not need a paired toggle.
- Boolean construction modes that already act as toggles (e.g. `input_transform: Literal["linear", "log2"]`). These are structural choices, not noise.
- The training-mode flag (`self.training`) carried by `nn.Module` itself; that is runtime, not config. Stochastic-rounding kernels in `neurox/common/quant.py` consume it directly — there is no separate `stochastic` override knob.

### Adding a new switch

1. Add the parameter field (scalar `float` or sub-config dataclass) to the config, no default.
2. Add a `validate_*` clause for the parameter (`_require_nonneg`, `_require_pos`, …).
3. Add a `bool` field to the module's `*Policy` dataclass, no default.
4. In the runtime, call the matching `apply_*` helper with `enabled=self.policy.<source>`.
5. Add the parameter value to the relevant `process/*.toml` if it is a PDK fact, otherwise to the chip TOML.
6. Update every call site that constructs the policy — there are no defaults, so the compiler / type checker will surface them.

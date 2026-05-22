# Config and Construction

This document records the current construction rules for NeuroX's device and analog / circuit hierarchy.

## Parameter classes

NeuroX distinguishes four parameter classes:

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

- `OpAmpTIAConfig` contains `nmos_cfg: NMOSConfig`.
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
- xbar tiles: `inst_shape: tuple[int, ...]` (per-instance multiplicity prefix; trailing `(col_num, w_digit_count, row_num)` is owned by the xbar's own cfg)
- xbar macros: `w_logical_shape: tuple[int, ...]` (operator-facing weight shape)

The shape is committed once at `__init__`, recorded on `self._inst_shape` (per the `FabricateMixin` contract), and the constructor records the profiler instance count from it. `fabricate()` then carries no arguments; it resamples mismatch at the already-bound shape. See [`fabrication_lifecycle.md`](fabrication_lifecycle.md) for the lifecycle.

## Family bases, concrete configs, and dispatch

Families with multiple concrete implementations use:

- one abstract base class
- one base config class
- one concrete config class per concrete implementation
- one concrete implementation class per concrete config

Each concrete implementation registers its config type on the family base via `RegistryMixin[type[<Family>Config], <Family>]`.

The base class exposes a family-specific `from_config(...)` classmethod that materialises the registry key from a config instance (`type(cfg)`) and instantiates the impl. `RegistryMixin` does not provide `from_config(...)`; it only provides:

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

Every frozen config inherits `ValidateMixin` (from `neurox/common/mixin/validate.py`) and defines exactly the same two methods, even when one or both are empty:

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

- Every frozen config **must inherit `ValidateMixin`** (transitively via a base config is fine). The mixin contributes only the runtime-check helpers; it does not define `__post_init__` or `validate`.
- `__post_init__` is defined on every config; it calls `self.validate()` and does nothing else. **Required even when** `validate` is empty.
- `validate()` is the single entry point for runtime validation. **Required on every config dataclass.**
- Multi-field configs split `validate()` into one `validate_<group>()` per logical group of fields. Group names follow field semantics (`validate_process` / `validate_temperature` / `validate_mismatch` / `validate_ppa` / `validate_noise` / `validate_topology` / `validate_geometry` / …). Use the standard name `validate_ppa` for the area / leakage / latency trio.
- Single-field configs may put the check directly inside `validate()` without a group method.
- Subclasses' `validate()` must start with `super().validate()` before calling their own group methods. This is the only way to chain validation up the inheritance tree.
- Empty / marker configs (e.g. `DACConfig`, `ADCConfig`) still define both methods explicitly, with `pass` as the `validate` body.

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

`neurox/common/load_dump.py:_build_value` enforces `Literal[...]` allowed sets at TOML / YAML load time. A field annotated `Literal["row_shared", "col_shared"]` rejects unknown values during deserialisation; `validate*` does not need to repeat the check.

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

# Config and Construction — Implementation

## Summary

The cross-cutting machinery that turns a tree of frozen `*Config` dataclasses into a tree of constructed modules: the four-class config layering (process / design / spec / runtime), config-type-keyed dispatch through `from_config`, owner-constructs-child ownership, `RegistryMixin` for per-family registries, `CircuitBase` / `CircuitConfig` as the standard electrical-circuit composition, and the `load_dump` (de)serializer that round-trips a config tree against TOML / YAML. This is software architecture with no physics-bearing reference counterpart; the device and circuit physics it carries are specified under [reference](../reference/README.md).

## Design decisions

- **`from_config` lives on the family base, never on the config or on `RegistryMixin`.** A config object must not be able to instantiate an arbitrary family, and a generic mixin cannot know a family's runtime arguments. The base owns the `type(config) → impl` mapping and exposes a family-specific signature, so different families carry different explicit runtime parameters (`XbarMacro.from_config` carries `ideal_xbar`; `TIA.from_config` does not). Rejected — one global `from_config` signature across all families: it would force a lowest-common-denominator argument list and let unrelated families silently diverge. Chosen — per-family factory with explicit, no-default parameters.

- **Dispatch is keyed on `type(config)`, not an `isinstance` ladder or a string tag in code.** Adding a concrete impl adds one `@Family.register_key(ConcreteConfig)` line and touches neither `from_config` nor any branch. The cost — one config class per impl — is paid once at design time. The string-discriminator variant (`RegistryMixin[Literal, Family]`, e.g. Transcoder keyed by `Encoding`) is the deliberate exception where the natural key is a value, not a type.

- **The config tree mirrors the ownership tree; the owner constructs the child.** If `A` owns `B`, `A`'s config carries `B`'s config as a field and `A.__init__` builds `B` directly. This makes ownership readable from the config structure and construction readable from the owning class, with no external factory closure deciding which child class is instantiated. Rejected — threaded factory closures (the pre-refactor design): ownership and construction drifted apart and design parameters leaked across layers. See [ADR-0001](../about/adr/ADR-0001-config-dispatch-and-owned-construction.md).

- **A device's *design* parameters belong to the owning circuit's config, not the device config.** Device configs hold only process and spec parameters; design parameters (`W__um`, `L__um`) are explicit `__init__` arguments. The reason is sharing: many device instances reuse one process config at different design sizes. When a circuit contains a device, that device's design parameters move onto the circuit config (`OpAmpTIAConfig` carries the pseudo-NMOS sizing because it is part of the TIA design) and the circuit constructs the device from them.

- **`config` and `policy` are split into two objects.** `config` is the immutable physical / design / spec reality (a PDK datum, reusable across studies); `policy` is the per-run study choice of which non-idealities to apply. Folding an `enable_*` flag onto the config conflated a noise source's *magnitude* with the *decision to model it*. The split lets calibration, training, and ablation runs pass different policies against one config, and makes the model-versus-experiment boundary visible in the constructor signature `Module(config=..., policy=...)`.

- **`CircuitBase` is a base class, not a loose mixin stack.** It fixes the canonical composition `FabricateMixin + ProfileMixin + nn.Module` plus `Generic[ConfigT]` so the typed config narrows in each leaf. Splitting static PPA into its own mixin separate from `ProfileMixin` would create a load-bearing implicit contract (the profile side reading PPA fields it does not declare) and force every leaf to compose mixins by hand; the single base avoids both. The typed area / leakage surface `CircuitBase` carries is the static-collection predicate for the profiler — see [`profiler.md`](common/profiler.md).

- **`ValidateMixin` is pure: it ships only check helpers, not `__post_init__` or `validate`.** Keeping the mixin free of the *when* (construction-time) decision means the project-wide rule that validation runs at `__post_init__ → validate()` lives in documentation and in each family base, not buried in mixin source. Whether the `__post_init__` entry point sits on the leaf or is inherited from a family base is a per-family simplicity choice.

- **`load_dump` is the sole format boundary.** Every config-loading entry point routes through it; no other module knows TOML / YAML. Its three private keys (`_neurox_type`, `_neurox_use`, `_neurox_use_preset`) are plain strings to any external parser. The loader is family-agnostic — it walks dataclass subclasses generically and has no table of which polymorphic families exist, so adding a family needs no loader change. See [`load_dump.md`](common/load_dump.md).

- **Policy files and config files are separate TOMLs, loaded by the same loader.** The immutable design lives in one file (`[macro]` / `[xbar]` section), the mutable runtime switches in another (`[policy]` section). `*Policy` dataclasses carry no field defaults and no Python factory (`all_off()` / `all_on()`); the all-off baseline is a single shared preset, `neurox/presets/policy/all_off.toml`, referenced via `_neurox_use_preset` and overridden switch-by-switch.

## Contracts & invariants

- **Config layering — process / design / spec / runtime.** Process: fixed by process selection. Design: freely chosen within one process. Spec: performance / statistics / bookkeeping (noise, mismatch, area, leakage, latency). Runtime: known only at run time (shape, active ADC mode, active bit width, per-call tensors). **Runtime parameters never appear on a config dataclass.** This is orthogonal to the Reference parameter-provenance taxonomy: layering answers "when is it fixed and who owns it", provenance answers "where does the number come from" (see [parameter_provenance](../reference/parameter_provenance.md)).

- **Every frozen config inherits `ValidateMixin`** (directly or transitively via `CircuitConfig`), **has a `validate()` on its inheritance chain** (empty bodies on marker bases are fine, so subclasses can `super().validate()`), and **is validated at construction** via a `__post_init__` that calls `self.validate()` and nothing else. A subclass `validate()` must open with `super().validate()` — the only way validation chains up the tree. A leaf that adds no checks does not override `validate()`; the chain already runs.

- **`validate*` encodes only what static analysis cannot.** Numeric bounds, monotonicity, length constraints, and cross-field relationships go in `validate*`; `isinstance`, `Literal` membership, and tensor shape / dtype do not — types are the annotation's job and `Literal` sets are enforced by the loader at deserialization. Do not duplicate a statically- or loader-checked property inside `validate*`. Multi-field configs split into one `validate_<group>()` per logical field group (use the standard name `validate_ppa` for the area / leakage / latency trio); single-field configs may inline the check.

- **A polymorphic family root inherits `RegistryMixin[KeyT, ImplT]` for its `type(config) → impl` dispatch.** The mixin supplies exactly two members the family relies on — `register_key(key)` (the class-definition-time binding decorator) and `_lookup_impl(key)` (the typed lookup) — and ships no `from_*` factory, so each family declares its own factory carrying the family's exact runtime-parameter signature. Config-class-keyed callers materialize the key at the call site as `cls._lookup_impl(type(config))`, keeping the mixin key-agnostic; discriminator families pass a string `Literal` key instead. The registry's own contract — per-family isolation, the rebind-refusing decorator, the import-driven completeness rule, and the broad on-disk storage type — lives in [`common/mixin/registry.md`](common/mixin/registry.md).

- **`from_config` takes explicit parameters only — no hidden defaults.** This prevents silent divergence between families and partially-applied construction rules.

- **`CircuitBase` is for electrical circuits only.** Devices (`RRAM` / `NMOS` / `Selector`) are physical primitives without per-op latency; they inherit `FabricateMixin + nn.Module` directly and their `*Config` does not inherit `CircuitConfig` — their static cost rolls up into the owning circuit's config. Macros (`XbarMacro` family) own zero silicon and only dispatch into children; they stay on `FabricateMixin + nn.Module + ProfileMixin + RegistryMixin` and surface PPA through their constituent circuits, not themselves.

- **Per-instance shape is committed once at `__init__`, recorded on `self._inst_shape`.** Leaf circuits and xbar tiles take `inst_shape`; xbar macros take the operator-facing `w_logical_shape`. `fabricate()` then carries no arguments — it resamples mismatch at the already-bound shape (the `FabricateMixin` contract, inherited through `CircuitBase`). The profiler derives instance count on demand from `_inst_shape`.

- **Policy structure mirrors config composition exactly.** Leaf modules carry flat `bool`-field policies; composites nest their sub-modules' leaf policies (a scheme-xbar policy holds its core policy plus the WL-DAC / clamp-driver / reference policies, ...). Polymorphic families have an empty abstract marker base policy plus one concrete policy per registered impl; the composite stores the abstract base type and the caller passes the concrete policy matching the config. Modules with no switches declare an empty marker policy for API uniformity. Each module stores its policy on the public `self.policy`.

- **`*Config` holds parameter values as fully-populated, non-Optional fields; `None` is forbidden.** Required scalars are `float`, multi-parameter distributions are required sub-config dataclasses. The apply / skip decision is never a config field — it is a `bool` on the paired `*Policy`. Runtime helpers in `neurox/common/nonideality.py` take a kw-only `*, enabled: bool` and short-circuit to pass-through when `False`, so call sites are single unbranched expressions with no `if`-gates and no `None` checks.

- **Field naming.** Config parameter field: `<source>__<unit>` when a physical unit exists, else `<source>`. Policy field: `<source>`, with no `enable_` prefix (the class role already implies enable). A category that sums several parameter fields gets one category-named policy field, not one per parameter.

- **Loader coercion contract.** `dataclass_from_dict` walks `typing.get_type_hints` and recurses element-wise: nested dataclasses from sub-tables; `list` / `tuple` / `set` / `dict` on inner types; fixed-length tuples enforce element count; `Enum` from `.value`; `Union` arms tried in order, first acceptor wins; **unknown keys raise `TypeError`** (a typo must not silently fall back to a default). `_neurox_type = "ConcreteConfig"` selects a polymorphic subclass by `__name__`; `Literal` sets are enforced here. `_neurox_use` resolves relative to the referencing file, `_neurox_use_preset` relative to `neurox/presets/`; inline keys override the referenced fragment, resolution is recursive and cycle-rejecting, and the two directives are mutually exclusive in one sub-table. Files under `neurox/presets/` must use `_neurox_use_preset` only — `_neurox_use` inside a preset subtree raises `ValueError`, keeping each preset's dependency graph closed inside the package.

## Performance & resources

N/A — construction and (de)serialization are one-time setup, off the per-VMM hot path. The compile- and memory-sensitive work lives in the per-subsystem internals (e.g. [xbar](xbar/README.md)).

## Gotchas

- **Adding a policy switch is a deliberate breaking change.** Because `*Policy` fields carry no defaults, a policy file or preset that omits a newly-added switch fails to load, and every call site that constructs the policy stops type-checking. This is intended: it forces every caller to declare a stance on the new source rather than inheriting a silent default. The six-step procedure (config field + `validate_*` clause + policy `bool` + `apply_*` call + preset / TOML value + call-site update) is in [`recipes.md`](../contributing/recipes.md).

- **Do not re-validate inner distribution configs.** Configs that hold a probability-distribution sub-config (`StuckAtFaultConfig`, `TelegraphConfig`, `LognormalConfig`, `GammaConfig`, and the `StateDependent*` variants in `neurox/common/nonideality.py`) must not re-check them in their own `validate*` — the inner config's `__post_init__` already validated on construction. Re-validating duplicates the rule and rots when the inner config changes.

- **`self.training` is not a policy switch.** The `nn.Module` training flag is runtime state, consumed directly by the stochastic-rounding kernels in `neurox/common/quant.py`; there is no `stochastic` override knob. Structural construction modes that already act as toggles (`input_transform: Literal["linear", "log2"]`) and non-noise numerical hyperparameters (softclip softness) follow the plain required-field rule and get no paired policy toggle.

- **`mypy` can lose `self.config` precision through `nn.Module.__getattr__`.** The `CircuitBase[VoltageDriverConfig]` generic parameter usually narrows `self.config`, but a parameterized `Generic` can lose precision through `nn.Module.__getattr__`; the leaf then needs a one-line class-level `config: VoltageDriverConfig` forward declaration alongside its buffer declarations.

## Known limitations

- **Per-op latency is not a `CircuitBase` contract.** Leaves with a dynamic profile model compute their own `latency__ns` and emit it via `_log_latency`; fixed-latency leaves read `self.config.latency_per_op__ns`; non-emitting leaves (`VoltageDriver`, `OpAmpTIA`) carry no per-op latency and fold their contribution into the owning circuit. There is no single base-level latency property to lean on.

- **No automated check that a policy tree matches its config tree.** The mirror between a composite's `*Policy` nesting and its `*Config` composition is a hand-maintained convention; a mismatched concrete policy passed for a polymorphic family is caught only at construction / type-check time, not by a structural assertion.

---

- **Reference**: none — this is software architecture; the physics it carries is specified under [reference](../reference/README.md).
- **Implementation**: `neurox/common/circuit.py`, `neurox/common/load_dump.py`, `neurox/common/mixin/registry.py`, `neurox/common/mixin/validate.py`, `neurox/common/nonideality.py`
- **Tests**: `tests/test_config_validation.py`, `tests/test_load_dump_use.py`, `tests/test_tool_config.py`
- **Decisions**: [ADR-0001](../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)

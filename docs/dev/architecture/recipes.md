# Recipes

Per-task checklists for adding new code to NeuroX. Each line points into an architecture document — the prose lives there, not here.

Every checklist applies in order. Skipping a step means the resulting module will fail a review.

## Add a new device

A device is an electrical primitive (transistor, memristor, wire, selector). Devices live under `neurox/device/`.

1. [config]      Define `<Name>Config` dataclass; process + spec only → [`config_and_construction.md` §Device layer](config_and_construction.md)
2. [validate]    `validate_<group>()` methods, called from `__post_init__` → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name>(nn.Module)` — devices do **not** inherit `ProfileMixin` → [`profiler_and_ppa.md` §Who profiles](profiler_and_ppa.md)
4. [init]        Design parameters go to `__init__` arguments, not the config → [`config_and_construction.md` §Device layer](config_and_construction.md)
5. [units]       Apply unit conversion from config units to tensor units inside `__init__` → [`physical_units.md` §Config units](physical_units.md)
6. [nominal]     Register `nominal_<name>__<unit>` buffers for design-stage values → [`state_holding.md` §Nominal value](state_holding.md)
7. [mixin]       Inherit `FabricateMixin`; override `_sample_fabricate_mismatch(self) -> None` to populate `<name>__<unit>` buffers from the nominals at `self._inst_shape` → [`fabrication_lifecycle.md` §Canonical signatures](fabrication_lifecycle.md)
8. [snapshot]    `snapshot(self, *, shape: tuple[int, ...]) -> <Name>Snapshot` returns a frozen dataclass of Tensors / nested Snapshots → [`state_holding.md` §Snapshot pattern](state_holding.md)
9. [solve]       `solve_dc(...) -> <Name>DCOP` is the primary DC entry; name and return suffix are reserved → [`naming_conventions.md` §Primary-method names](naming_conventions.md)
10. [export]     `neurox/device/__init__.py` exports `<Name>`, `<Name>Config`, `<Name>Snapshot`, `<Name>DCOP` → [`config_and_construction.md` §Export rule](config_and_construction.md)
11. [doc]        Write `docs/dev/modules/device/<name>.md` describing the device's current responsibility and protocol surface → [`code_style.md` §Module docs](code_style.md)

## Add a new leaf circuit (no family polymorphism)

A leaf circuit is a non-polymorphic analog or digital block (Driver, SwitchCap, AnalogMux, …, or one of the digital primitives). Lives under `neurox/analog/` or `neurox/digital/`.

1. [config]      Define `<Name>Config(CircuitConfig)`; design + spec + member-config fields. `CircuitConfig` provides area + leakage; subclass adds its own — including a `latency_per_op__ns: float` field if the leaf emits dynamic events with a fixed per-op latency → [`config_and_construction.md` §Circuit layer](config_and_construction.md), [`modules/common/circuit.md`](../modules/common/circuit.md)
2. [validate]    `validate_<group>()` methods, called from `__post_init__`; in `validate_ppa` `super().validate_ppa()` then `_require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")` (and any added energy fields) → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name>(CircuitBase[<Name>Config])` — `CircuitBase` composes `FabricateMixin + ProfileMixin + nn.Module + Generic[ConfigT]` → [`profiler_and_ppa.md` §Who profiles](profiler_and_ppa.md), [`modules/common/circuit.md`](../modules/common/circuit.md)
4. [init]        Standard family signature `__init__(self, *, config, name, inst_shape, dtype, T__K)` — every kwarg is required, no defaults anywhere in the physical layer; convert config units in `__init__`; call `super().__init__(config=config, name=name, inst_shape=inst_shape)` first → [`physical_units.md` §Config units](physical_units.md), [`code_style.md` §Physical-layer no defaults](code_style.md), [`modules/common/circuit.md`](../modules/common/circuit.md)
5. [nominal]     Register `nominal_<name>__<unit>` buffers (Tensor) or store Python scalars for design-stage values → [`state_holding.md` §Nominal value](state_holding.md)
6. [mixin]       Override `_sample_fabricate_mismatch(self) -> None` to refresh actual-value buffers from nominals + static mismatch at `self._inst_shape`; use buffer reassignment, not `register_buffer` → [`fabrication_lifecycle.md` §Canonical signatures](fabrication_lifecycle.md), [`code_style.md` §FabricateMixin and buffer reassignment](code_style.md)
7. [PPA]         `CircuitBase` already provides `area_per_inst__um2 / leakage_per_inst__uW` properties reading from `self.config`. Per-op latency is leaf-defined: fixed-latency leaves read `self.config.latency_per_op__ns`; parametric leaves derive it from runtime parameters → [`profiler_and_ppa.md` §Where per-op latency comes from](profiler_and_ppa.md)
8. [primary]     Implement the primary method (`convert`, `transport`, `sample_and_accumulate`, `operate`, …); end by building energy and latency tensors and calling `self._log_dynamic_energy(energy)` and `self._log_latency(latency)` as two independent emissions (skip either one when the leaf has no contribution for that quantity). `latency` is `per_op_latency__ns × serial_op_count` packaged as a tensor; `per_op_latency__ns` reads from config (fixed) or comes from runtime parameters (parametric). Compute `serial_op_count` from the forward tensor's shape (minus inst dims and per-circuit trailing dims) → [`naming_conventions.md` §Primary-method names](naming_conventions.md), [`profiler_and_ppa.md` §Serial-op count](profiler_and_ppa.md)
9. [compile]     Primary methods on the compile path; do not self-decorate with `@torch.compile` → [`compile_policy.md` §Where the compile boundary lives](compile_policy.md)
10. [export]     The owning package's `__init__.py` exports `<Name>`, `<Name>Config` (+ any `*Snapshot` / `*DCOP` if produced) → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
11. [doc]        Write `docs/dev/modules/<path>/<name>.md`; include `Compile-path: yes/no` on each primary method → [`compile_policy.md` §Per-method declaration](compile_policy.md)

## Add a new concrete member of an existing family

Adding a new ADC, DAC, TIA, or ReadOut implementation. Lives under the family's subpackage.

1. [config]      Define `<Name><Family>Config(<Family>Config)` extending the family base config → [`config_and_construction.md` §Family bases](config_and_construction.md)
2. [validate]    `validate_<group>()` methods in the new config → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name><Family>(<Family>)`; add a `config: <Name><Family>Config` class-level forward declaration; register via `@<Family>.register_key(<Name><Family>Config)` → [`config_and_construction.md` §Family bases](config_and_construction.md), [`ADR-0001`](../adr/ADR-0001-config-dispatch-and-owned-construction.md)
4. [init]        Use the family-wide signature (`config, name, inst_shape, dtype, T__K` + family-specific extras like `ideal_xbar`); no defaults on any kwarg; call `super().__init__(...)` → [`config_and_construction.md` §Family-wide init](config_and_construction.md), [`code_style.md` §Physical-layer no defaults](code_style.md)
5. [mixin]       Override `_sample_fabricate_mismatch(self)` for owned static mismatch; `fabricate()` is inherited and auto-cascades to children → [`fabrication_lifecycle.md` §Canonical signatures](fabrication_lifecycle.md)
6. [primary]     Implement the family primary method (`convert`, `solve_dc`, `readout`, …); end with `self._log_dynamic_energy(energy_tensor)` and `self._log_latency(latency_tensor)` (skip either when this impl has no contribution for that quantity) → [`naming_conventions.md` §Primary-method names](naming_conventions.md), [`profiler_and_ppa.md` §`_log_dynamic_energy`](profiler_and_ppa.md)
7. [PPA]         PPA accessors are inherited from `CircuitBase`; override only when latency / area / leakage are derived from non-config fields → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md)
8. [export]      The family's `__init__.py` exports `<Name><Family>`, `<Name><Family>Config`, any `*Snapshot` / `*DCOP` types → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
9. [doc]         Write `docs/dev/modules/<path>/<name>.md`; cross-reference the family base in `See also:` → [`code_style.md` §Module docs](code_style.md)

## Add a new family

Creating a polymorphic-family namespace (sibling of `ADC`, `DAC`, `TIA`, `ReadOut`). Rare.

1. [adr]         Write an ADR explaining why the new family is needed and what alternatives were rejected → [`docs/dev/adr/README.md`](../adr/README.md)
2. [base-config] `<Family>Config(CircuitConfig)` frozen dataclass (inherits area + leakage from `CircuitConfig`; add family-specific fields, including a `latency_per_op__ns: float` field if every family impl has fixed per-op latency, otherwise leave latency to each concrete `*Config`) → [`config_and_construction.md` §Family bases](config_and_construction.md), [`modules/common/circuit.md`](../modules/common/circuit.md)
3. [base-class]  `class <Family>(CircuitBase[<Family>Config], RegistryMixin[type["<Family>Config"], "<Family>"])` with `from_config(...)` classmethod, family-wide `__init__` signature, and abstract primary methods → [`config_and_construction.md` §Family bases](config_and_construction.md), [`ADR-0001`](../adr/ADR-0001-config-dispatch-and-owned-construction.md)
4. [contracts]   Declare the primary method (`convert` / `solve_dc` / `readout` / …). Static-PPA accessors (`area_per_inst__um2` / `leakage_per_inst__uW`) are inherited from `CircuitBase` — do not redeclare. Per-op latency is leaf-defined inside the primary method via `_log_latency(latency_tensor)` (paired with `_log_dynamic_energy(energy_tensor)` when the impl also emits energy). `fabricate()` is provided by `FabricateMixin` and subclasses override `_sample_fabricate_mismatch` only → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md), [`naming_conventions.md` §Primary-method names](naming_conventions.md)
5. [docs-base]   Write `docs/dev/modules/<path>/base.md` (or family `README.md`) describing the protocol surface in abstract terms (no specific consumer names) → [`code_style.md` §Documentation dependency direction](code_style.md)
6. [first-impl]  Add at least one concrete impl (see "Add a new concrete member" recipe above)

## Add a new value-domain primitive (slicer, transcoder)

Slicer is an abstract base with direct concrete subclasses (callers instantiate the concrete class by name). Transcoder uses `RegistryMixin[Encoding, "Transcoder"]` so concrete subclasses self-register on a string discriminator and `Transcoder.create(encoding, ...)` dispatches.

1. [base]        Abstract class (`Slicer`, `Transcoder`) declares only the externally observable surface: the primary method and any abstract `@property` (e.g. `value_range`, `slice_radix`, `slice_weights` on `Slicer`). No shared `__init__` or stored state if subclass init signatures diverge. → [`mapping.md`](mapping.md)
2. [class]       Concrete subclass; constructor takes only the parameters the subclass itself consumes. Structural defaults of a particular subclass stay internal — do not surface them as caller-side kwargs. The per-call signature carries only the input tensor. → [`mapping.md`](mapping.md)
3. [register]    For a registry-dispatched family (Transcoder): add `@<Family>.register_key("<discriminator>")` on the concrete subclass; for direct-instantiation families (Slicer): omit this step → [`docs/dev/modules/common/registry_dispatch.md`](../modules/common/registry_dispatch.md)
4. [primary]     Implement the primary method (`slice`, `encode` / `decode`) → [`naming_conventions.md` §Primary-method names](naming_conventions.md)
5. [output]      Return raw `Tensor` for one-tensor returns; reserve `*Plan` / `*Result` dataclasses for the case when a method must return multiple runtime-computed tensors that have no useful identity as instance state. Static geometry stays on the producing class as `@property` → [`code_style.md` §Property vs method](code_style.md), [`naming_conventions.md` §Class suffixes](naming_conventions.md)
6. [export]      Owning package `__init__.py` exports the concrete class + any `*Plan` / `*Result` types in its public type annotations → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
7. [doc]         Write `docs/dev/modules/mapper/.../<name>.md` → [`code_style.md` §Module docs](code_style.md)

## Add a user-side operator (out of core)

`nn.Module`-level replacements for stock PyTorch layers (`nn.Linear`, `nn.Conv2d`, …) wrapping an `XbarMacro` live in the **user's repository or in `example/`**, not in the core `neurox/` tree. The core public surface stops at `neurox.macro`; anything that pairs a macro with a PyTorch layer, manages QAT observers, or handles model rewriting is application-layer code.

See the `example/lenet/quant.py` and `example/bert/quant.py` files for a current reference of the macro-wrapping pattern.

# Recipes

Per-task checklists for adding new code to NeuroX. Each line points into an architecture document — the prose lives there, not here.

Every checklist applies in order. Skipping a step means the resulting module will fail a review.

## Add a new device

A device is an electrical primitive (transistor, memristor, wire, selector). Devices live under `neurox/device/`.

1. [config]      Define `<Name>Config` dataclass; process + spec only → [`config_and_construction.md` §Device layer](config_and_construction.md)
2. [validate]    `validate_<group>()` methods, called from `__post_init__` → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name>(nn.Module)` — devices do **not** inherit `ProfiledModule` → [`profiler_and_ppa.md` §Who profiles](profiler_and_ppa.md)
4. [init]        Design parameters go to `__init__` arguments, not the config → [`config_and_construction.md` §Device layer](config_and_construction.md)
5. [units]       Apply unit conversion from config units to tensor units inside `__init__` → [`physical_units.md` §Config units](physical_units.md)
6. [nominal]     Register `nominal_<name>__<unit>` buffers for design-stage values → [`state_holding.md` §Nominal value](state_holding.md)
7. [fabricate]   `fabricate(self, shape: tuple[int, ...]) -> None` populates `<name>__<unit>` buffers from the nominals + static mismatch → [`fabrication_lifecycle.md` §Canonical signature](fabrication_lifecycle.md)
8. [snapshot]    `snapshot(self, *, shape: tuple[int, ...]) -> <Name>Snapshot` returns a frozen dataclass of Tensors / nested Snapshots → [`state_holding.md` §Snapshot pattern](state_holding.md)
9. [solve]       `solve_dc(...) -> <Name>DCOP` is the primary DC entry; name and return suffix are reserved → [`naming_conventions.md` §Primary-method names](naming_conventions.md)
10. [export]     `neurox/device/__init__.py` exports `<Name>`, `<Name>Config`, `<Name>Snapshot`, `<Name>DCOP` → [`config_and_construction.md` §Export rule](config_and_construction.md)
11. [doc]        Write `docs/dev/modules/device/<name>.md` describing the device's current responsibility and protocol surface → [`code_style.md` §Module docs](code_style.md)

## Add a new leaf circuit (no family polymorphism)

A leaf circuit is a non-polymorphic analog or digital block (Decoder, Driver, SwitchCap, AnalogMux, …, or one of the digital primitives). Lives under `neurox/analog/` or `neurox/digital/`.

1. [config]      Define `<Name>Config` dataclass; design + spec + member-config fields → [`config_and_construction.md` §Circuit layer](config_and_construction.md)
2. [validate]    `validate_<group>()` methods, called from `__post_init__` → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name>(nn.Module, ProfiledModule)` → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md)
4. [init]        Standard family signature `__init__(self, *, cfg, name, T__K, dtype)` — every kwarg is required, no defaults anywhere in the physical layer; convert config units in `__init__` → [`physical_units.md` §Config units](physical_units.md), [`code_style.md` §Physical-layer no defaults](code_style.md)
5. [nominal]     Register `nominal_<name>__<unit>` buffers (Tensor) or store Python scalars for design-stage values → [`state_holding.md` §Nominal value](state_holding.md)
6. [fabricate]   `fabricate(self, shape: tuple[int, ...]) -> None` populates actual-value buffers from nominals + static mismatch, then `self._record_inst_count(shape)` → [`fabrication_lifecycle.md` §Canonical signature](fabrication_lifecycle.md), [`profiler_and_ppa.md` §`_record_inst_count`](profiler_and_ppa.md)
7. [PPA]         `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns` properties (or method when latency depends on runtime args) → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md)
8. [primary]     Implement the primary method (`convert`, `transport`, `sample_and_accumulate`, `drive`, `operate`, …); end with `self._log_dynamic(e__fJ, latency__ns)` → [`naming_conventions.md` §Primary-method names](naming_conventions.md), [`profiler_and_ppa.md` §Dynamic energy](profiler_and_ppa.md)
9. [compile]     Primary methods on the compile path; do not self-decorate with `@torch.compile` → [`compile_policy.md` §Where the compile boundary lives](compile_policy.md)
10. [export]     The owning package's `__init__.py` exports `<Name>`, `<Name>Config` (+ any `*Snapshot` / `*DCOP` if produced) → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
11. [doc]        Write `docs/dev/modules/<path>/<name>.md`; include `Compile-path: yes/no` on each primary method → [`compile_policy.md` §Per-method declaration](compile_policy.md)

## Add a new concrete member of an existing family

Adding a new ADC, DAC, TIA, or ReadOut implementation. Lives under the family's subpackage.

1. [config]      Define `<Name><Family>Config(<Family>Config)` extending the family base config → [`config_and_construction.md` §Family bases](config_and_construction.md)
2. [validate]    `validate_<group>()` methods in the new config → [`code_style.md` §Configuration validation](code_style.md)
3. [class]       `class <Name><Family>(<Family>)`; register via `@<Family>.register_key(<Name><Family>Config)` → [`config_and_construction.md` §Family bases](config_and_construction.md), [`ADR-0001`](docs/dev/adr/ADR-0001-config-dispatch-and-owned-construction.md)
4. [init]        Use the family-wide signature (`cfg, name, T__K, dtype` + family-specific extras like `ideal_xbar`); no defaults on any kwarg; call `super().__init__(...)` → [`config_and_construction.md` §Family-wide init](config_and_construction.md), [`code_style.md` §Physical-layer no defaults](code_style.md)
5. [fabricate]   Implement `fabricate(self, shape: tuple[int, ...])`; end with `self._record_inst_count(shape)`. Family-specific structural facts go through `__init__`, not `fabricate` → [`fabrication_lifecycle.md` §Canonical signature](fabrication_lifecycle.md)
6. [primary]     Implement the family primary method (`convert`, `solve_dc`, `readout`, …) → [`naming_conventions.md` §Primary-method names](naming_conventions.md)
7. [PPA]         Implement `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns` → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md)
8. [export]      The family's `__init__.py` exports `<Name><Family>`, `<Name><Family>Config`, any `*Snapshot` / `*DCOP` types → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
9. [doc]         Write `docs/dev/modules/<path>/<name>.md`; cross-reference the family base in `See also:` → [`code_style.md` §Module docs](code_style.md)

## Add a new family

Creating a polymorphic-family namespace (sibling of `ADC`, `DAC`, `TIA`, `ReadOut`). Rare.

1. [adr]         Write an ADR explaining why the new family is needed and what alternatives were rejected → [`docs/dev/adr/README.md`](docs/dev/adr/README.md)
2. [base-config] `<Family>Config` frozen dataclass (often empty, a marker for the dispatch registry) → [`config_and_construction.md` §Family bases](config_and_construction.md)
3. [base-class]  `class <Family>(nn.Module, ProfiledModule, RegistryDispatchMixin[type["<Family>Config"], "<Family>"], ABC)` with `from_config(...)` classmethod, family-wide `__init__` signature, and abstract primary methods → [`config_and_construction.md` §Family bases](config_and_construction.md), [`ADR-0001`](docs/dev/adr/ADR-0001-config-dispatch-and-owned-construction.md)
4. [contracts]   Declare abstract `fabricate`, the primary method (`convert` / `solve_dc` / `readout` / …), `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns` → [`profiler_and_ppa.md` §Required interface](profiler_and_ppa.md), [`naming_conventions.md` §Primary-method names](naming_conventions.md)
5. [docs-base]   Write `docs/dev/modules/<path>/base.md` (or family `README.md`) describing the protocol surface in abstract terms (no specific consumer names) → [`code_style.md` §Documentation dependency direction](code_style.md)
6. [first-impl]  Add at least one concrete impl (see "Add a new concrete member" recipe above)

## Add a new value-domain primitive (slicer, transcoder)

Slicer is an abstract base with direct concrete subclasses (callers instantiate the concrete class by name). Transcoder uses `RegistryDispatchMixin[Encoding, "Transcoder"]` so concrete subclasses self-register on a string discriminator and `Transcoder.create(encoding, ...)` dispatches.

1. [base]        Abstract class (`Slicer`, `Transcoder`) declares only the externally observable surface: the primary method and any abstract `@property` (e.g. `value_range`, `slice_radix`, `slice_weights` on `Slicer`). No shared `__init__` or stored state if subclass init signatures diverge. → [`mapping.md`](mapping.md)
2. [class]       Concrete subclass; constructor takes only the parameters the subclass itself consumes. Structural defaults of a particular subclass stay internal — do not surface them as caller-side kwargs. The per-call signature carries only the input tensor. → [`mapping.md`](mapping.md)
3. [register]    For a registry-dispatched family (Transcoder): add `@<Family>.register_key("<discriminator>")` on the concrete subclass; for direct-instantiation families (Slicer): omit this step → [`docs/dev/modules/common/registry_dispatch.md`](docs/dev/modules/common/registry_dispatch.md)
4. [primary]     Implement the primary method (`slice`, `encode` / `decode`) → [`naming_conventions.md` §Primary-method names](naming_conventions.md)
5. [output]      Return raw `Tensor` for one-tensor returns; reserve `*Plan` / `*Result` dataclasses for the case when a method must return multiple runtime-computed tensors that have no useful identity as instance state. Static geometry stays on the producing class as `@property` → [`code_style.md` §Property vs method](code_style.md), [`naming_conventions.md` §Class suffixes](naming_conventions.md)
6. [export]      Owning package `__init__.py` exports the concrete class + any `*Plan` / `*Result` types in its public type annotations → [`naming_conventions.md` §Class suffixes](naming_conventions.md)
7. [doc]         Write `docs/dev/modules/mapper/.../<name>.md` → [`code_style.md` §Module docs](code_style.md)

## Add a new operator

An operator is a `nn.Module`-level replacement for a stock PyTorch layer (`nn.Linear`, `nn.Conv2d`, …). Lives under `neurox/operator/<kind>/`.

1. [class]       `class <Name>(NeuroxOperator)` → [`code_style.md` §Module docs](code_style.md)
2. [init]        Constructor takes a `macro: NeuroxMacroQuantMatMul` plus the layer's geometric parameters; register integer-weight / scale / zero-point buffers
3. [from_torch]  Provide `from_torch(cls, module, macro, name)` classmethod for in-place replacement
4. [forward]     `forward(self, input) -> Tensor` runs the int matmul pipeline through the macro; do not decorate with `@torch.compile` (the macro handles compilation) → [`compile_policy.md` §Where the compile boundary lives](compile_policy.md)
5. [fabricate]   `fabricate(self) -> None` validates and programs the macro from the loaded int weights
6. [export]      `neurox/operator/__init__.py` exports the new class
7. [doc]         Write `docs/dev/modules/operator/<kind>/<name>.md`

# Recipes

Task routing and checklists for core NeuroX changes.

## Common checklist

Every core change starts here:

1. Identify the task shape below and the owning base class, mixin, shared subsystem, or package surface.
2. Update or create the [Reference](../reference/README.md) spec first when the physical, mathematical, numerical, or public semantic contract changes.
3. Update or create the [Internals](../internals/README.md) document when implementation design, lifecycle, shape / dtype / buffer contracts, ownership, performance, compile behavior, or package surface changes.
4. Apply content-placement, dependency, and single-source rules through [organizing_principles](../conventions/organizing_principles.md), code rules through [code_style](../conventions/code_style.md), and documentation text and format rules through [prose_style](../conventions/prose_style.md) and [markdown_style](../conventions/markdown_style.md).
5. Implement through the relevant base-class or mixin contract. Do not re-state that contract in the leaf implementation.
6. Update package exports and public API documentation when the import surface changes; follow [package_surface](../internals/package_surface.md).
7. Add or update focused tests and validation evidence.
8. Run the relevant [workflow](workflow.md) quality gates, including `make docs-build` for documentation or link changes.

Small internal refactors that do not change behavior or contracts may skip Reference updates, but they still update Internals when they change design rationale or maintenance constraints.

## Add a pure electrical primitive

Applies to foundational electrical models such as devices and other primitive I/V elements whose silicon rolls up to an owning block, so they self-account no static PPA.

Use the common checklist, then:

- Define `*Config` and, when runtime switches exist, `*Policy`; declare no per-instance area / leakage fields, as the owner budgets them.
- Implement the primitive as a `ModuleBase` leaf that overrides `reports_static_ppa: ClassVar[bool] = False`, so the profiler's static walk skips it and its area and leakage are counted once at the owner.
- Follow the physical-state and lifecycle contracts in [physical_state](../internals/physical_state.md).
- Provide `snapshot` and / or `solve_dc` only when the primitive owns that runtime concept.
- Export the public class and role dataclasses from the owning package.
- Test physical equations, validation failures, snapshot behavior, and DC solve behavior where applicable.

## Add a profiled circuit leaf

Applies to analog and digital leaf circuits that own their own silicon and emit PPA profile events.

Use the common checklist, then:

- Define `*Config` (extending the subsystem config base — e.g. `AnalogConfig` plus its own `area_per_inst__um2` / `leakage_per_inst__uW`, or `DigitalConfig`) and its `*Policy`, plus validation groups.
- Use the construction contract in [common/base](../internals/common/base.md) and the `ProfileMixin` emitter contract: set the bare per-instance PPA data (`_area_per_inst__um2` / `_leakage_per_inst__uW`) in `__init__`, which `area__um2` / `leakage__uW` scale by `inst_count`.
- Implement the family or leaf primary method defined by its base class.
- Emit dynamic energy and latency only for quantities this leaf owns.
- Keep fixed latency in config; derive parametric latency inside the primary method when required.
- Test shape contract, dtype behavior, PPA emissions, and edge cases for the primary method.

## Add a registry family

Applies when adding a dispatchable abstract family.

Use the common checklist, then:

- Define base `*Config` / `*Policy` role types only when the family owns those concepts.
- Define the abstract base surface and `from_config` dispatch through `RegistryMixin` or a documented equivalent.
- Document the family contract in the family base Reference / Internals pages before adding concrete members.
- Keep shared method docstrings on the abstract declaration.
- Add at least one concrete member or document why the base is introduced ahead of implementations.
- Test dispatch, validation, abstract contract enforcement, and public exports.

## Add a concrete registry member

Applies when extending an existing dispatch family.

Use the common checklist, then:

- Extend the family config / policy types according to the family base contract.
- Register the implementation with the correct key type.
- Implement only the behavior owned by the concrete member.
- Do not repeat base-class contracts in the concrete docs; document concrete model and implementation differences.
- Export the public class and role dataclasses from the family package.
- Test dispatch from config, primary behavior, validation, and any concrete non-idealities.

## Add a composite owned-construction block

Applies to modules that own child modules, nested config / policy, layout transforms, programmed state, or profile aggregation.

Use the common checklist, then:

- Define ownership: which children are constructed directly, which are created through `from_config`, and which runtime context each receives.
- Document shape / layout contracts in Reference when they are part of the model and in Internals when they are implementation layout; use [organizing_principles](../conventions/organizing_principles.md) for the cross-cutting convention.
- Keep public construction and propagation rules in [config_and_policy](../internals/config_and_policy.md).
- Put detailed lifecycle and state behavior in [physical_state](../internals/physical_state.md).
- Test owned construction, `fabricate` cascade, `program`, primary execution, shape transforms, and profile aggregation.

## Add a numerical solver or compiled algorithm leaf

Applies to numerical algorithms, solver leaves, chunking helpers, and compile / eager boundaries.

Use the common checklist, then:

- Document the mathematical method in Reference and the implementation constraints in Internals.
- State shape, dtype, convergence, memory, and compile-safety contracts explicitly.
- Do not force the implementation into a `ModuleBase` leaf or other hardware-module pattern unless it truly owns that role.
- Keep hot paths free of Python-state mutation and dynamic behavior forbidden by the compile contract.
- Test residuals, convergence / fixed-iteration behavior, shape edge cases, dtype behavior, and chunk reassembly.

## Add a value-domain primitive

Applies to pure tensor mappings such as encoding, transcoding, and slicing.

Use the common checklist, then:

- Define a small abstract surface: primary transform methods and static geometry / range properties.
- Use direct construction or registry dispatch according to the family contract.
- Return raw tensors for single-tensor results.
- Keep static geometry as properties on the producing object.
- Test round trips, value ranges, shape transforms, and invalid inputs.

## Add a policy switch

Applies when adding a runtime non-ideality toggle to a module's `*Policy`.

Adding a policy switch is a deliberate breaking change: `*Policy` fields carry no defaults, so a policy file or preset that omits the new switch fails to load and every call site that constructs the policy stops type-checking, forcing each caller to declare a stance on the new source.

Use the common checklist, then apply the six-step procedure:

1. Add the `*Config` magnitude field, when the source has a paired parameter.
2. Add the `validate_*` clause that bounds that field.
3. Add the `bool` field to the paired `*Policy`, with no `enable_` prefix.
4. Add the `apply_*` call gated by the policy `bool` on the runtime path.
5. Set the value in every preset and config / policy TOML the switch reaches.
6. Update every call site that constructs the policy.

## Modify shared infrastructure

Applies to mixins, `ModuleBase` / config / policy bases, profiler, non-ideality helpers, quantization helpers, load / dump, preset schema, and other high-impact common mechanisms.

Use the common checklist, then:

- Identify all callers and downstream contracts before implementation.
- Update the public contract document that owns the mechanism.
- Prefer backward-compatible migration when possible; otherwise document the breaking change and update recipes / conventions that route to the mechanism.
- Add tests at the shared contract level and at least one representative downstream use.

# Recipes

Task routing and checklists for core NeuroX changes.

## Common checklist

Every core change starts here:

1. Identify the task shape below and the owning base class, mixin, shared subsystem, or package surface.
2. Update or create the [Reference](../reference/README.md) spec first when the physical, mathematical, numerical, or public semantic contract changes.
3. Route the software record by the scope of what changed. A changed caller-facing or subclass-facing contract — shape, dtype, state, lifecycle, ownership, or the obligations a subclass must meet — is written in the docstring of the symbol that owns and enforces it. A changed contract that spans components, which no single symbol can hold, goes to its system design page, written per [writing_system_design](writing_system_design.md). A purely local refactor changes code, comments, and tests only, and creates no document.
4. Apply content-placement, dependency, and single-source rules through [organizing_principles](../conventions/organizing_principles.md), code rules through [code_style](../conventions/code_style.md), and documentation text and format rules through [prose_style](../conventions/prose_style.md) and [markdown_style](../conventions/markdown_style.md).
5. Implement through the relevant base-class or mixin contract. Do not re-state that contract in the leaf implementation.
6. Update package exports and public API documentation when the import surface changes, and keep the new module inside the package imports dispatch depends on — see [construction](../system_design/construction.md).
7. Add or update focused tests and validation evidence.
8. Run the relevant [workflow](workflow.md) quality gates.

## Writing tests

These hold for every test the checklist adds:

- Write each test's config and policy by hand, stating in the test the values its assertions depend on; do not reach for a preset or a production TOML to obtain them.
- Assert laws, not numbers — invariants, monotonicity, scaling and limiting relations, and boundary behavior — so that recalibrating a physical parameter does not rewrite the suite. A literal number belongs in an assertion only when it is itself the specification, such as an analytic closed form or an exact-integer result.
- Keep one test file per module under test, so the guard for a symbol is found from that symbol's module path.

## Introduce a simulation primitive

A standalone primitive represents an independent, reusable physical or circuit concept with its own behavioral contract. Neither the number of current callers nor the complexity of its formula decides whether the concept deserves that boundary. Pure arithmetic stays in the composing owner, while a block modelled only as static PPA or data-independent per-operation cost stays a seat under [code_style](../conventions/code_style.md) rather than becoming a class.

## Add a pure electrical primitive

Applies to foundational electrical models such as devices and other primitive I/V elements whose silicon rolls up to an owning block, so they self-account no static PPA.

Use the common checklist, then:

- Use the family's config and policy role types. Add specialized descendants only when the primitive introduces fields or a distinct dispatch identity; do not create empty per-class types merely to match its name. The common bases supply their frozen, keyword-only dataclass representation. Declare no per-instance area / leakage fields, as the owner budgets them.
- Implement the primitive as a `ModuleBase` leaf that stays a non-reporter: its area and leakage are counted once at the owner, and it emits no dynamic event of its own. Declare the profile-target class variable as [code_style](../conventions/code_style.md) prescribes.
- Follow the physical-state and lifecycle contracts in [physical_state](../system_design/physical_state.md).
- Provide `snapshot` and / or `solve_dc` only when the primitive owns that runtime concept.
- Export the public class and role dataclasses from the owning package.
- Test physical equations, validation failures, snapshot behavior, and DC solve behavior where applicable.

## Add a profiled circuit leaf

Applies to analog and digital leaf circuits that own their own silicon and emit PPA profile events.

Use the common checklist, then:

- Use or extend the subsystem config and policy bases. Declare new role types only for new fields or a distinct dispatch identity, never merely to mirror the module class name; do not repeat dataclass decorators, and put new domain checks in `validate()`.
- Use the `ModuleBase` construction contract and the `ProfileMixin` emitter contract: implement the per-instance PPA properties the mixin requires, which `area__um2` / `leakage__uW` scale by `inst_count`.
- Implement the family or leaf primary method defined by its base class.
- Emit dynamic energy only for quantities this leaf owns, as a tensor at the billed layout; the profiler owns the reduction.
- Declare `latency__ns` on the concrete circuit that owns a propagation or conversion delay, or on the narrowest family base when every member owns the same timing contract. `ModuleBase` does not impose a timing interface.
- At a functional boundary, `latency__ns` covers one complete public operation, including every serial axis owned below that boundary. Compose it explicitly from known execution structure; do not infer it by traversing the module tree or by treating every child latency as additive.
- Keep paper measurement periods, external clock periods, and leakage-integration windows in validation code unless the runtime model explicitly implements their scheduling semantics.
- Test shape contract, dtype behavior, PPA emissions, and edge cases for the primary method.

## Add a registry family

Applies when adding a dispatchable abstract family.

Use the common checklist, then:

- Define the config and policy role types that form the family dispatch key; an empty marker is appropriate when its type identity distinguishes the family even though it has no fields.
- Define the abstract base surface and `from_config` dispatch through `RegistryMixin` or a documented equivalent.
- State the family's extension contract in the abstract base's docstring before adding concrete members, and add the family Reference document when the shared science is substantial.
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
- Document shape / layout contracts in Reference when they are part of the model and in the owning class docstring when they are implementation layout; use [organizing_principles](../conventions/organizing_principles.md) for the carrier rule.
- Follow the owner-constructs-child and config-propagation protocol in [construction](../system_design/construction.md).
- Follow the lifecycle and state-ownership contracts in [physical_state](../system_design/physical_state.md), and document this composite's own lifecycle behavior in its class docstring.
- Test owned construction, `fabricate` cascade, `program`, primary execution, shape transforms, and profile aggregation.

## Add a numerical solver or compiled algorithm leaf

Applies to numerical algorithms, solver leaves, chunking helpers, and compile / eager boundaries.

Use the common checklist, then:

- Document the mathematical method in Reference; the implementation constraints belong to the docstrings of the symbols that impose them.
- State shape, dtype, convergence, memory, and compile-safety contracts explicitly.
- Do not force the implementation into a `ModuleBase` leaf or other hardware-module pattern unless it truly owns that role.
- Keep hot paths free of Python-state mutation and dynamic behavior forbidden by the [compile contract](../system_design/compile.md).
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

1. Add the `*Config` magnitude field when the source has one.
2. Add the field's bounds directly to the owning class's `validate()` method.
3. Add the `bool` field to `*Policy`, with no `enable_` prefix.
4. Add the `apply_*` call gated by the policy `bool` on the runtime path.
5. Set the value in every preset and config / policy TOML the switch reaches.
6. Update every call site that constructs the policy.

## Modify shared infrastructure

Applies to mixins, `ModuleBase` / config / policy bases, profiler, non-ideality helpers, quantization helpers, load / dump, preset schema, and other high-impact common mechanisms.

Use the common checklist, then:

- Identify all callers and downstream contracts before implementation.
- Update the public contract document that owns the mechanism.
- Add tests at the shared contract level and at least one representative downstream use.

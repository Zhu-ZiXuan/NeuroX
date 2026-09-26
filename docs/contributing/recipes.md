# Change recipes

Choose the row matching your change. Keep interface contracts in their owning docstrings and scientific models in Reference, following the [documentation ownership rules](../conventions/organizing_principles.md).

| Change | Update | Verify |
| --- | --- | --- |
| Electrical primitive | Model equations, parameter fields, and the electrical interface | Independent numerical oracles and relevant physical limits |
| Profiled circuit | Local hardware costs, energy events, and operation timing | Ownership, enabled events, and interaction with profiling |
| Registry family | Base-class extension contract, config-policy dispatch, and exports | Registration and representative construction |
| Concrete implementation | Model-specific behavior, restrictions, and exported types | The behavior added beyond the family contract |
| Composite unit | Child ownership, mapping, state lifecycle, and timing composition | End-to-end numerical results and hardware accounting |
| Numerical algorithm | Mathematical method and callback, shape, dtype, and convergence contracts | Independent solutions, termination, and chunk reassembly |
| Encoding or slicing | Representable domains, positional weights, and reconstruction | Round trips, range boundaries, and composition |
| Shared infrastructure | Owning contracts and affected callers | Shared behavior plus a representative downstream use |

For every change:

1. Identify the responsible abstraction and read its calling or extension contract.
2. Update the code, relevant tests, and affected documentation together. A local refactor needs no new design page.
3. Update exports and import-time registration when the public surface changes.
4. Select checks using [test guidance](writing_tests.md) and run the [contribution checks](../../CONTRIBUTING.md#quality-gates).

## Adding a policy field

Config and policy fields have no defaults, so a new field affects constructors and configuration files. Add its validation, implement the selected behavior, and update every affected preset, run file, and caller. Document the field's meaning beside its declaration and any new physical source in the model reference.

# Writing tests

Test ownership, inputs, assertions, and organization for NeuroX changes.

## Test selection

Apply two questions to every proposed or existing test:

1. **Own responsibility:** Assuming every dependency fully satisfies its contract, what meaningful defect in the implementation under test can this test detect?
2. **Additional coverage:** What does this test protect that existing tests of the same responsibility do not already adequately cover?

Dependencies include third-party libraries and project components, whether imported, inherited, or supplied by a caller. Trusting them does not assume the implementation uses or combines them correctly; integration tests qualify by the composition errors they detect. Thin wrappers do not warrant dedicated tests: this includes forwarding arguments, exposing properties, or applying a dependency operation through shared field traversal. Generated field assignment likewise needs no test. Test a composition only when it introduces meaningful logic beyond that delegation.

Keep tests with substantive answers to both questions. Before removing overlapping coverage, preserve any unique assertions in the surviving tests; similar inputs alone do not establish redundancy.

Do not test a direct implementation of a simple fixed physical formula by repeating its expression, elementary arithmetic properties, or dependency-provided broadcasting. Replaying the implementation's physical steps in the same order is likewise not an independent oracle. Keep physical tests when they expose nontrivial implementation errors through an independent method or meaningful composition, such as finite differences for analytic derivatives, conservation residuals for numerical solvers, or sign, phase, and positional-weight assembly across components.

Simple, explicit, locally readable defensive `assert` or `raise` checks do not warrant dedicated mirror tests, even when testing their owning base class or infrastructure directly. Merely guarding against deletion of the check or changes to its exception wording is insufficient. Keep a validation test only when it protects meaningful logic beyond the straightforward predicate, such as nontrivial condition combinations, validation-hook execution, or failure ordering and state preservation.

## Inputs and assertions

- Write each test's config and policy by hand, stating in the test the values its assertions depend on; do not reach for a preset or a production TOML to obtain them.
- Assert laws, not numbers — invariants, monotonicity, scaling and limiting relations, and boundary behavior — so that recalibrating a physical parameter does not rewrite the suite. A literal number belongs in an assertion only when it is itself the specification, such as an analytic closed form or an exact-integer result.
- Assert observable API or extension contracts, not internal representations or call sequences. A behavior-preserving refactor should not require rewriting its tests.
- An observable consequence of the current implementation is not automatically a contract. Do not freeze object reuse, storage sharing, cache retention, or incidental random-generator activity. Test results at the operation that owns them; test state stability only when the required semantics independently demand it.
- Prefer stable behavior boundaries, but directly test a private function when it owns nontrivial algorithmic logic that deserves isolated verification. Private visibility alone neither requires nor forbids a test; storage layout and helper decomposition are not behavioral contracts.
- Enforce simple input constraints and output invariants at their owning runtime boundary when every call needs that protection. Do not mechanically replace removed tests with checks already guaranteed by dependencies. Prefer explicit exceptions for mandatory validation and account for tensor-scan and synchronization costs.
- Use independent numerical oracles. Test doubles control dependencies or observe contractual interactions; they need not replace dependencies merely to isolate a layer.
- Inspect compiler output, optimization-dependent resource use, and workload-specific numerical convergence as development diagnostics, not hard test assertions.

## Organization

Keep project-specific static rule checks in `tests/rules/`. Organize behavior tests by the implementation that owns the behavior, following its package path; using a dependency in a fixture does not make that dependency the test's owner. Cross-component tests live with the composing owner.

Use one test file for a small module. When a module needs several substantial behavior groups, collect them in one module-named directory and split by behavior, such as mapping, timing, or energy accounting. Avoid parallel smoke, foundation, and regression suites that repeat those groups. Keep helpers local to their consumers; `tests/utils/` holds only genuinely shared test support.

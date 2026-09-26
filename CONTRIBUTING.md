# Contributing to NeuroX

Prepare a checkout, verify a focused change, and submit it for review.

## Prepare a source checkout

Run `uv sync --locked` from the repository root to create the development environment. `pyproject.toml` is the authority for supported Python and dependency ranges and for dependency groups; `uv.lock` records the resolved development environment. A change that requires an API outside a declared range updates that range deliberately and refreshes the lockfile in the same branch.

Use the Makefile as the authority for repository-level development tasks. Run `make help` to discover the current targets rather than relying on a copied target list.

## Branch and PR flow

1. Branch from `main` with a short, descriptive name such as `codex/fix_profile_layout`.
2. Keep commits focused: one logical change per commit.
3. Update code, tests, and affected documentation in the same branch.
4. Run the local quality gates from the repository root.
5. Open a PR against `main` and describe what changed and why.
6. Address review feedback with follow-up commits on the same branch.

Do not commit directly to `main`. Do not bundle unrelated changes into one PR.

## Quality gates

Use `make help` to select the current formatting, lint, type-checking, test, and documentation targets. Run the smallest relevant check while iterating, then broaden verification in proportion to the change before review. A documentation or link change requires the documentation target; a code change requires the focused behavior tests plus every broader gate affected by its surface.

Run the formatting target before staging final code changes because it edits files.

Ruff `E` and `W` rules (pycodestyle errors and warnings) are enforced; fix every `E` and `W` finding the branch produces. mypy is a helper, not a gate: use `mypy_report.log` to triage the findings relevant to the change rather than to chase a clean run.

Add a custom test under `tests/rules/` only for a repository-specific invariant that Python, Ruff, mypy, MkDocs, and the existing test suite cannot reliably express. Do not reproduce a configured tool's check with reflection or AST machinery.

A PR that changes numerical or physical behavior must also add or run the validation that covers the change. Extend or add the tests for the affected equation, device, circuit, or model and run them; pick the validation that matches the behavior rather than a fixed device- or GPU-specific command.

## Focused validation

Run subsets directly when useful:

```bash
uv run pytest tests/path/to/test_file.py
uv run pytest -k some_keyword
```

Validation campaigns are local end-to-end resources, not universal PR gates. Use `make help` to discover them and select the task whose documented behavior covers the change.

## Issues and focused contributions

Report bugs with a minimal reproducer, the dependency versions, the device, and the expected and observed results. For a scientific discrepancy, include the model assumptions and parameter provenance. Describe the problem and validation in a PR; keep unrelated refactors separate.

Use [change recipes](docs/contributing/recipes.md) to select the relevant implementation, test, and documentation work. [Authoring guidance](docs/contributing/README.md) covers the detailed writing tasks.

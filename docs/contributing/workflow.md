# Contribution workflow

Process for taking a focused change from a branch to review. Content ownership and task routing live in [recipes](recipes.md).

## Branch and PR flow

1. Branch from `main` with a short, descriptive `snake_case` name.
2. Keep commits focused: one logical change per commit.
3. Update code, tests, and affected documentation in the same branch.
4. Run the local quality gates from the repository root.
5. Open a PR against `main` and describe what changed and why.
6. Address review feedback with follow-up commits on the same branch.

Do not commit directly to `main`. Do not bundle unrelated changes into one PR.

## Quality gates

| Gate | Command | Notes |
|---|---|---|
| Format | `make format` | Runs Ruff format and auto-fixable lint rules; rewrites files. |
| Lint | `make lint` | Runs Ruff and writes `ruff_report.log`. |
| Types | `make check` | Runs mypy and writes `mypy_report.log`. |
| Tests | `make test` | Runs pytest over `tests` by default. |
| Docs | `make docs-build` | Builds the documentation site. |

Run `make format` before staging final code changes because it edits files.

Ruff `E` and `W` rules (pycodestyle errors and warnings) are enforced; fix every `E` and `W` finding the branch produces. mypy is a helper: per project policy not every warning must be zero, so use `mypy_report.log` to triage findings relevant to the change rather than to chase a fully clean run.

Run `make docs-build` for documentation changes and for code changes that update doc links or API surfaces. Use `make docs-serve` only for local preview.

A PR that changes numerical or physical behavior must also add or run the validation that covers the change. Extend or add the tests for the affected equation, device, circuit, or model and run them; pick the validation that matches the behavior rather than a fixed device- or GPU-specific command.

## Focused validation

Run subsets directly when useful:

```text
pytest tests/path/to/test_file.py
pytest -k some_keyword
```

Example training / evaluation Make targets are local end-to-end resources, not required PR gates. Use `make help` to discover them when you need that workflow.

## Routing

Use [recipes](recipes.md) to decide which code, tests, Reference documents, Internals documents, and package-surface documents must change with the branch.

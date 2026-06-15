# Contribution Workflow

How to take a change from a branch to a merged PR, and the gates it must clear on the way.

## Branch and PR flow

1. Branch off `main`. Use a short, descriptive `snake_case` name (`add_rram_drift`, `fix_solver_offset`).
2. Make small, focused commits. Keep one logical change per commit.
3. Run the quality gates locally (below) until they pass.
4. Push the branch and open a PR against `main`. Describe what changed and why; link the issue if one exists.
5. Address review feedback by pushing follow-up commits to the same branch.

Do not commit to `main` directly. Do not bundle unrelated changes into one PR.

## Quality gates

Run these before opening a PR, and again after addressing review. Each target is defined in the `Makefile`; run them from the repository root.

| Gate | Command | Checks |
|---|---|---|
| Format | `make format` | `ruff` formatter plus auto-fixable lint rules — rewrites files in place |
| Lint | `make lint` | `ruff` linter (runs `make format` first), writes `ruff_report.log` |
| Types | `make check` | `mypy` static analysis over `neurox`, `example`, `tests`, writes `mypy_report.log` |
| Tests | `make test` | `pytest` over `tests` |

`make format` edits files, so run it first and stage the result. `make lint` and `make check` report findings without changing code; they are advisory helpers, so use judgment rather than chasing every warning to zero. `make test` must pass before you open a PR.

A `.pre-commit-config.yaml` wires `ruff format`, `ruff check --fix`, `ruff check`, and `mypy` as pre-commit hooks. Install it with `pre-commit install` to run the lint and type gates automatically on every commit.

## Running validation

Run the test suite directly when you want a subset:

```text
make test                 # full suite
pytest tests/path/to/test_file.py
pytest -k some_keyword
```

The bundled examples double as end-to-end validation of the simulator. Each model exposes `train`, `hat` (hardware-aware QAT), and `eval` targets, for example:

```text
make eval-lenet MAX_SAMPLES=1000
make eval-bert
```

`make help` lists every target with its description. Before running any GPU task, check that a CUDA device has free memory and is idle. Run on `DEVICE=cuda:0` by default; override `DEVICE` to pick another. If a run hits out-of-memory, lower `BATCH_SIZE` first.

To preview documentation changes, run `make docs-serve` (live reload) or `make docs-build` (one-shot build).

## Expectations for new code

New code is expected to follow the project conventions:

- [code_style](code_style.md) — docstrings, inline comments, shape annotations, dependency direction, unit-suffix encoding, dtype policy.
- [naming_conventions](naming_conventions.md) — physical-quantity suffixes, dataclass role suffixes, method and profiler names.
- [recipes](recipes.md) — per-task checklists (add a device, a leaf circuit, a family member, a new family).

When a change alters behavior, update the affected documents in the same PR: see [writing_reference_docs](writing_reference_docs.md) for spec-level docs and [writing_internals_docs](writing_internals_docs.md) for implementation notes. Match the prose and character rules in [doc_style](doc_style.md).

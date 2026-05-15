# Development Principles

All contributors are expected to follow these principles when working on NeuroX.

## Language and Runtime

- **Python** 3.11 or later.
- **PyTorch** 2.9 or later. Use the latest API — do not add backward-compatibility shims for older versions.
- Assume `torch.compile` is applied to callers. Write code that traces cleanly under TorchInductor: no in-place mutation, no `tensor.new_*` constructors, no boolean indexing on data-dependent masks, no `.item()` inside compiled regions.

## Code Style

- **Formatter:** `ruff format`. Run `make format` before every commit.
- **Linter:** `ruff check`. Run `make lint` to auto-fix and report. Treat remaining warnings as helpers, not blockers — fix what makes sense.
- **Type checker:** `mypy`. Run `make check`. Not every warning needs fixing, but public APIs must have correct annotations.
- **Type hints** are required on all function signatures (parameters and return type).

## Naming and File Layout

- Filenames: `snake_case.py`, matching the primary class (`class ShiftAdder` → `shift_adder.py`).
- One primary class per file. Supporting types (configs, NamedTuples, private helpers) may share the file.
- Subpackage `__init__.py` exports the public API. External code imports from the package, not from private modules.
- Physical variables include units in their names: `g__mS`, `v__V`, `i__mA`, `energy__nJ`, `area__um2`, `latency__ns`.

## Configuration

- All tunable parameters live in frozen `@dataclass` configs, never as magic constants in source code.
- Device configs use `NamedTuple` for noise parameters (one tuple per noise source).
- Default parameter sets live in `neurox/config/` and are imported by example scripts.

## `torch.compile` Practices

- **Do not scatter `@torch.compile` on every small function.** Place it at the level where fusion gives the biggest win — typically the Newton residual check, the Newton step, the read-noise pipeline, and the digital aggregation.
- **Never nest `@torch.compile` regions** — an inner compiled function becomes a graph break for the outer one, preventing fusion and potentially forcing intermediate materialization.
- Functions that should **inline** into a compiled caller (e.g., `_cell_current__mA`, noise `apply_*` helpers) must **not** carry their own `@torch.compile` decorator.
- Functions that serve as compile **entry points** should have a docstring that explains *why* they are compiled and what fusion benefit it provides.
- Set `torch._dynamo.config.cache_size_limit` high enough (64+) when many layers with different shapes share the same compiled function.

## Git Workflow

1. **Master branch** is always releasable.
2. **Feature branches** are cut from master: `feature/your-feature-name`.
3. **Commits** are small and focused. Message format: `[module] description`, e.g., `[xbar] add damped Newton-Raphson solver`.
4. **Pre-commit:** run `make format` then `make lint`. Fix errors before pushing.
5. **Pull requests** require passing CI and at least one code review.

## Testing

- Framework: `pytest`.
- Coverage: `pytest-cov`. Core modules (device, xbar, macro) target 90%+ line coverage.
- Test layout mirrors source: `test/unit/`, `test/integration/`.
- Tests must run on both CPU and CUDA. Use `@pytest.mark.skipif` for GPU-only tests.

## Docstring Standard

NeuroX uses **Google-style** docstrings, rendered by `mkdocstrings` into the documentation site.

### Module docstring (top of every `.py` file)

The module docstring is the primary documentation entry point. It should cover, when applicable:

- **Purpose:** what the module models and why it exists.
- **Physical model:** the equations, physical variables (with units), and key assumptions.
- **Noise and non-idealities:** which noise sources are modeled and how (distribution, parameters).
- **Solver or algorithm:** if the module contains a numerical solver, describe the method (e.g., Newton-Raphson with damping), convergence criterion, and any tricks (Thomas algorithm, Schur complement).
- **Memory optimization:** if the module uses serialization, `@torch.compile` fusion, or leading-dimension chunking, explain the strategy and why it is necessary.
- **Tensor layout:** for modules that handle multi-dimensional broadcast tensors, document the dim ordering and which dims are config-fixed vs. layer-varying.

Keep it concise. State facts; do not repeat the code.

### Class docstring

One paragraph describing the class's role. List constructor arguments only if they are not obvious from the config.

### Method docstring

Required for all public methods. Use `Args:`, `Returns:`, and optionally `Raises:` sections. Include tensor shapes in `Shape: [...]` comments when the shape is non-trivial.

### Docstring examples

```python
"""Noise helpers for device and circuit models.

Each noise type is described by its own ``NamedTuple`` config.  The
``apply_*`` functions are branch-free elementwise expressions that
trace cleanly inside a ``@torch.compile`` caller — no boolean
indexing, no in-place writes.
"""
```

```python
def load_g__mS(self, state: Tensor, ...) -> Tensor:
    """Map discrete state indices to noisy conductance values.

    Applies programming gamma noise, temporal drift, and stuck-at
    faults in sequence.  Read noise is *not* applied here — it is
    sampled per access in ``read_g__mS``.

    Args:
        state: Integer state indices in ``[0, num_states - 1]``.
        t_elapsed: Time since programming in seconds.

    Returns:
        Conductance in mS, clamped to ``[g_min, g_max]``.
    """
```

## Documentation Build

- Tool: `mkdocs` with `mkdocs-material` theme.
- API reference: auto-generated from docstrings via `mkdocstrings`.
- Build locally: `mkdocs serve` from the repo root.
- All narrative docs live in `doc/` and are referenced from `mkdocs.yml`.

# NeuroX Development Docs

This directory (`docs/dev/`) holds the long-lived internal documentation for NeuroX. It is organised into two layers:

- `architecture/` records current cross-cutting rules that apply across multiple modules.
- `adr/` stores architectural decision records: why the project chose a particular design and which alternatives were rejected.

Per-module design notes live in the sibling `docs/modules/` tree at the docs root, recording the current design of specific packages and files.

Code docstrings should stay short and local. They should describe the current interface contract, tensor shape contract, units, invariants, and other facts a reader needs while using the code. Longer design discussion and historical rationale belongs here instead.

Start with these documents:

- `docs/dev/architecture/code_style.md` for code docstring, comment, and shape-annotation rules.
- `docs/dev/architecture/config_and_construction.md` for config, `from_config`, and ownership rules.
- `docs/dev/architecture/state_holding.md` for fabricated-state ownership.
- `docs/dev/architecture/fabrication_lifecycle.md` for the `__init__` / `fabricate` / `snapshot` / `forward` split.
- `docs/dev/architecture/mapping.md` for the macro / mapper / tiler / slicer / xbar layering.
- `roadmap.md` for the implemented vs. planned feature matrix.

Module-level docs live under `docs/modules/`, mirroring the `neurox/` source tree one-to-one. ADRs live under `adr/`.

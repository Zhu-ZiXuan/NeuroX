# NeuroX Development Docs

This directory holds the long-lived internal documentation for NeuroX.

Development documentation is split into three layers:

- `architecture/` records current cross-cutting rules that apply across multiple modules.
- `modules/` mirrors the source tree and records the current design of specific packages and files.
- `adr/` stores architectural decision records: why the project chose a particular design and which alternatives were rejected.

Code docstrings should stay short and local. They should describe the current interface contract, tensor shape contract, units, invariants, and other facts a reader needs while using the code. Longer design discussion and historical rationale belongs here instead.

Start with these documents:

- `architecture/code_style.md` for code docstring, comment, and shape-annotation rules.
- `architecture/config_and_construction.md` for config, `from_config`, and ownership rules.
- `architecture/state_holding.md` for fabricated-state ownership.
- `architecture/fabrication_lifecycle.md` for the `__init__` / `fabricate` / `snapshot` / `forward` split.
- `architecture/mapping.md` for the macro / mapper / tiler / slicer / xbar layering.
- `roadmap.md` for the implemented vs. planned feature matrix.

Module-level docs live under `modules/`, mirroring the `neurox/` source tree one-to-one. ADRs live under `adr/`.

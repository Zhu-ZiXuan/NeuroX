# NeuroX

NeuroX is a PyTorch-based simulator for RRAM compute-in-memory accelerators. It connects device and circuit models to crossbar arrays, CIM macros, and neural-network operators so that one hardware configuration can be evaluated for functional accuracy and static PPA.

The supported library surface ends at `neurox.architecture.unit`. Model rewriting, training, and end-to-end evaluation pipelines live under `example/` as application code rather than stable API.

## Get started

Follow [Get started](docs/get_started/README.md) to prepare a supported environment and run a bundled workflow.

## Documentation

- [Guides](docs/guides/README.md) — end-to-end workflows grouped by task.
- [Scientific reference](docs/reference/README.md) — modeled components, equations, assumptions, and limitations.
- [Validation](docs/validation/README.md) — evidence connecting implementations to published designs.
- [API](docs/api/README.md) — the supported Python surface and configuration format.
- [System design](docs/system_design/README.md) — contracts that span components.
- [Contributing](docs/contributing/README.md) — development and documentation workflow.

NeuroX is released under the [MIT License](LICENSE).

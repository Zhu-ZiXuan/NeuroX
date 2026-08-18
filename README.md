# NeuroX

NeuroX is a PyTorch-based simulator for RRAM compute-in-memory accelerators. It connects device and circuit models to crossbar arrays, CIM macros, and neural-network operators so that one hardware configuration can be evaluated for functional accuracy and static PPA.

The supported library surface ends at `neurox.architecture.unit`. Model rewriting, training, and end-to-end evaluation pipelines live under `example/` as application code rather than stable API.

## Get started

NeuroX requires Python 3.12 or later. To prepare a source checkout with the dependencies used by the bundled examples:

```bash
uv sync --extra demo
```

Follow the [algorithm-engineer workflow](docs/guides/algorithm_engineer/workflow.md) to train and evaluate the LeNet or BERT pipeline.

## Documentation

- [Guides](docs/guides/README.md) — end-to-end workflows grouped by task.
- [Scientific reference](docs/reference/README.md) — modeled components, equations, assumptions, and limitations.
- [Validation](docs/validation/README.md) — evidence connecting implementations to published designs.
- [API](docs/api/README.md) — the supported Python surface and configuration format.
- [System design](docs/system_design/README.md) — contracts that span components.
- [Contributing](docs/contributing/README.md) — development and documentation workflow.

NeuroX is released under the [MIT License](LICENSE).

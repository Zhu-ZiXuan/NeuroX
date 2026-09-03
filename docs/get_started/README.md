# Get started

Installation and the route to a first runnable example.

## Install

The supported Python and dependency ranges live in `pyproject.toml`. From a source checkout, install the library, development tools, and dependencies used by the bundled examples:

```bash
uv sync --extra demo
```

Optional dependency groups for other workflows are declared in `pyproject.toml` and can be selected with `uv sync --extra <name>`.

## Run a bundled workflow

Follow the [algorithm-engineer workflow](../guides/algorithm_engineer/workflow.md) to train or evaluate the bundled LeNet and BERT pipelines and inspect their accuracy and PPA output. Contributors who are preparing a development change continue with the [contribution workflow](../contributing/workflow.md).

# NeuroX documentation

NeuroX models RRAM compute-in-memory hardware from devices and circuits through integer linear and convolution operators. Use it to investigate numerical behavior and modeled power, performance, and area (PPA).

## Use the library

Start with the [quickstart](get_started/README.md), which needs no checkpoints or datasets. Continue with the [operator workflow](guides/algorithm_engineer/workflow.md) and [configuration guide](api/configuration.md). The [API reference](api/README.md) provides calling contracts from source docstrings.

## Understand the results

The [scientific reference](reference/README.md) explains the modeled equations and assumptions. [Validation](validation/README.md) describes how implementations are checked. Read the [scope and limitations](about/scope_limitations.md) and [PPA accounting](system_design/ppa_accounting.md) before interpreting a report as a hardware estimate.

## Extend the library

The [extension API](api/extensions.md) documents the bases and hooks used by new implementations. [System explanations](system_design/README.md) connect responsibilities across components.

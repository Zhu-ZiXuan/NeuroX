# Interfaces and versioning

## Interface audiences

The application surface provides construction, lifecycle operations, integer compute units, profiling, and reporting. [Application APIs](../api/python.md) and [unit APIs](../api/units.md) document those calls.

[Extension interfaces](../api/extensions.md) and [circuit components](../api/components.md) support new implementations and lower-level experiments. A documented extension hook carries the obligations stated by its base even when its name begins with an underscore. Other private helpers are implementation details.

## Compatibility

The package version and supported dependency ranges are declared in `pyproject.toml`. NeuroX is pre-1.0; application and extension interfaces may change between releases. Record the release or commit used for reproducible research, and review interface changes when upgrading. Documenting an interface does not imply a post-1.0 stability guarantee.

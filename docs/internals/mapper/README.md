# mapper — Implementation

How the value-domain mapping layer is built. Spec: [reference/mapper](../../reference/mapper/README.md); this side covers only what the code cannot tell you - the ABC observable surfaces, the registry dispatch, and the construction contracts.

- [transcoder/](transcoder/README.md) — the `Transcoder` ABC, registry-backed encoding dispatch, and the per-encoding implementation contracts.
- [xbar/](xbar/README.md) — the `Slicer` ABC and the value-decomposition implementations.

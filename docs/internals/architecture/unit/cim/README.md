# CimUnit family

How the `CimUnit` family is built.

- [base](base.md) — `CimUnit` registry root plus the `EngineBackedCimUnit` intermediate that owns and delegates to a `CimEngine`.
- [linear](linear.md) — `LinearCimUnit`: the linear-operator engine-backed unit.
- [conv2d](conv2d.md) — `Conv2dCimUnit`: the conv2d-operator engine-backed unit (Toeplitz lowering).
- [engine/](engine/README.md) — the `CimEngine` registry family: the tile-build helper, the sub-phase plane machinery, the chunk-and-pad primitive, and the per-variant organize/aggregate shape pipelines.
- [slicer/](slicer/README.md) — the `Slicer` ABC observable surface and the value-decomposition implementations.

The registry's substrate-free members — `IdealLinearUnit` and `IdealConv2dUnit` — live beside their operator ABCs one level up and are documented on those pages.

# Unit

How the unit layer is built. This side covers only what the code cannot tell you.

- [base](base.md) — the `UnitBase` root ABC: the matmul-shaped lowering template and its three hook seams, the `int_bias` slot mechanics, and why the contract carries no PPA surface.
- [linear](linear.md) — the `LinearUnit` operator ABC and the `IdealLinearUnit` exact-integer reference leaf.
- [conv2d](conv2d.md) — the `Conv2dUnit` operator ABC (geometry-parameterized conv seams) and the `IdealConv2dUnit` int64 im2col reference leaf.
- [cim/](cim/README.md) — the `CimUnit` registry family: registry dispatch, the engine-backed delegation, and the per-variant organize/aggregate shape pipelines.

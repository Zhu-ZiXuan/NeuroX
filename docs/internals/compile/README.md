# Compilation

How NeuroX runs under `torch.compile`: the contracts every compile-friendly function obeys, and the schemes that address the one hard compile problem in the library.

- [contracts](contracts.md) — the dynamo-safety rules every compile-friendly function obeys, and the compile-friendly-by-default expectation for module docs
- [scheme_a_regional](scheme_a_regional.md) — Active: regional compilation of the `solve_dc` leaf, the boundary map, and the A/B/C scheme comparison
- [scheme_b_deobjectified](scheme_b_deobjectified.md) — Deferred: the de-objectified free-function solver, a hardening of scheme A
- [scheme_c_custom_op](scheme_c_custom_op.md) — Deferred: the xbar-read custom op for a `fullgraph=True` parent

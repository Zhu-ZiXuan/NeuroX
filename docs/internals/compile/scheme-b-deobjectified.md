# Scheme B — De-objectified Solver

**Status: Deferred — not implemented.** A hardening of [scheme A](scheme-a-regional.md), recorded as future work. Adopt when any of: cross-instance reuse proves fragile on the real path (per-layer recompilation observed); a predictable, object-independent compile cache key is wanted; or as the prerequisite functionalization for [scheme C](scheme-c-custom-op.md).

## The problem it solves

Scheme A compiles `solve_dc` as a bound method, but a stateless one: it reaches through `self` only for its scalar numerical config (the iteration counts on `self.config`, plus the class-level Newton damping caps). The cell (with its device models), the two clamp drivers, and all of their per-call snaps arrive as `solve_dc` arguments, not through `self`. The compile-cache concern is the device buffers reachable *through* those argument actors: whether the many same-geometry macro layers share **one** compiled leaf or each cold-compile their own depends on the compile cache key staying a function of tensor shape/dtype/device plus scalar config, and **not** of object identity (the solver object, a buffer instance carried by a cell or driver). In scheme A that property rides on an implicit, version-dependent mechanism (dynamo lifting nn.Module buffers as shape-guarded inputs). If it ever fails, every layer recompiles — a build-time cost linear in layer count, with no error to signal it.

De-objectification removes the dependence on that implicit mechanism: the compiled region becomes a pure function whose cache key is, by construction, only its code plus the meta of its tensor arguments plus its scalar parameters.

## The PyTorch mechanism it leans on

> May change across versions — re-confirm before relying on an edge.

The compile cache key of a module-level function is determined by the function code, the tensor argument shapes/dtypes/devices, and the concrete values of scalar (int/float/bool) arguments — none of which is an object identity. Passing all dynamic state as explicit tensor arguments, and all scalar configuration as small type-stable values, keeps the key object-independent. The pattern mirrors the relationship between a stateful `nn.Linear` (holds parameters) and the stateless `nn.functional.linear` (pure computation).

## Implementation form

The concrete solver math moves to a module-level function; the method becomes a thin adapter.

```python
@dataclass(frozen=True)
class NestedParallelRailSolverCompileParams:
    n_outer: int
    n_inner: int
    max_outer_step__V: float
    max_inner_step__V: float
    # every scalar the method reads through self: the iteration counts off
    # self.config plus the class-level Newton damping caps

@torch.compile(dynamic=False)
def solve_nested_chunk(
    *,
    bl_segment_g__uS: Tensor,
    sl_segment_g__uS: Tensor,
    cell_snap: XbarCellSnap,
    bl_driver_snap: OpAmpTIASnap,
    sl_driver_snap: VoltageDriverSnap,
    params: NestedParallelRailSolverCompileParams,
) -> tuple[Tensor, ...]:
    ...
```

The sketch flattens the per-call actors (the cell, the two clamp drivers) down to their snaps for the functionalization step: the cell condenses its own RRAM / NMOS device branch into `cell_snap` (the free function never sees raw `rram_snap` / `nmos_snap`), and each clamp driver's fabricated state arrives as its own driver snap. The device behaviour and the clamp solves are recovered by calling each actor's stateless solve helper on the matching snap, exactly as `NestedParallelRailSolver.solve_dc` does today. `NestedParallelRailSolver.solve_dc` then becomes the adapter that only: reads its scalar config off `self`, takes the cell, the two clamp drivers, and their snaps from the call, calls the free function, and reassembles the returned tuple into `SolverDCOP`. If the snap dataclasses themselves cause recompiles, the next step is to expand them into plain `Tensor` arguments so the signature is fully tensor-and-scalar.

## Trade-offs

- **vs. scheme A.** Gains a compile cache key that is object-independent by construction, so cross-instance reuse is guaranteed rather than incidental, and the "every layer recompiles" failure mode is structurally impossible; also lays the functional foundation [scheme C](scheme-c-custom-op.md) requires. Loses code simplicity: the solver gains a wrapper/core split and a longer surface, and the adapter's packing must be kept in sync with the free function's signature.
- **Returning a dataclass vs a tuple.** Returning `SolverDCOP` keeps call sites unchanged but ties the compiled region to a PyTree-stable structure; returning a bare tuple and reassembling in the adapter is more robust to PyTree edges at the cost of an explicit repack.
- **Dataclass snap args vs pure tensors.** Dataclass args stay close to current code; pure-tensor args maximize cache stability but lengthen the signature. Start with dataclass snaps and only flatten if recompiles are observed.

## Performance and resources (theoretical)

Runtime is unchanged — the arithmetic is identical, only its argument plumbing differs. The gain is a compile signature that is provably one graph per (shape, config) across all instances, removing the per-layer cold-compile risk of scheme A.

## Risks and failure modes

- **Snap PyTree instability.** A snap field whose Python type or container structure varies call-to-call reintroduces recompiles; the point of the scheme is lost unless the argument structure is type-stable.
- **Wrapper/core drift.** The adapter and the free function encode the same math in two places; an edit to one without the other is a silent correctness hazard. The split must be maintained deliberately.
- **Scalar config must be type-stable.** Passing an object or a varying Python type where a scalar is expected re-keys the graph.

## Open questions

- How far to de-objectify: stop at dataclass snaps, or flatten everything to plain tensors.
- The remainder-chunk decision (accept an extra graph vs pad-to-full-chunk) is shared with scheme A and can be settled here.

## See also

- [scheme A](scheme-a-regional.md), [scheme C](scheme-c-custom-op.md), [contracts](contracts.md)

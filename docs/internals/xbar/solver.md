# Crossbar Solver Primitives — Implementation

## Summary

`neurox/xbar/solver.py` holds the topology-agnostic numerical primitives reused across crossbar IR-drop solvers: batched tridiagonal and block-tridiagonal linear solves, the per-column wire KCL residual and driver-current builders, and an elementwise-Jacobian-diagonal helper. Every helper takes its tensors as plain arguments and owns no topology, no device handles, and no solver framework. The solver framework itself (base class, registry, DCOP / residual containers, Newton loops) lives with its topology family — see [_1t1r/solver](_1t1r/solver.md). This document covers the implementation choices for the shared primitives; the physics and the Newton method that consume them are specified by the owning topology.

## Design decisions

- **Topology-agnostic numerical primitives, kept out of any one Newton loop.** These are pure linear-algebra and KCL builders parameterised entirely by their tensor arguments. Placing them in a shared module separates them from any single topology's solver framework, so multiple topology families can reuse the same Thomas / block-Thomas kernels and the same residual diagnostics without depending on each other.
- **Shared residual / driver-current builders for consistent diagnostics.** `col_wire_kcl_residual` and `col_driver_current` are the single source for per-column wire-ladder KCL and boundary current. Sharing them keeps residual diagnostics numerically identical across every solver that builds the same wire ladder, rather than each solver re-deriving the stamping.
- **No in-place mutation.** Both Thomas sweeps (`solve_tridiagonal`) and the block-Thomas sweeps (`solve_block_tridiagonal`) accumulate intermediate results to stay compile-friendly (no in-place mutation). In-place tensor writes break `torch.compile` graph tracing; pure-functional accumulation lets Inductor fuse the per-step elementwise ops across the unrolled loop.
- **Fixed shape convention for the block solver (no `dim` arg).** `solve_block_tridiagonal` fixes the $N$ axis as third-to-last for the block tensors (`[..., N, B, B]`) and second-to-last for the RHS (`[..., N, B]`); callers transpose at the call site if their layout differs. The scalar `solve_tridiagonal` instead takes an explicit `dim` because its slices are rank-uniform with the RHS and an axis argument costs nothing.
- **One block solve per step, not two.** The block forward sweep stacks the super-diagonal block and the RHS column into a single right-hand side and issues one `torch.linalg.solve` per row instead of two, then splits the result back into the coupling factor $C_k$ and the reduced RHS $d_k$.

## Contracts & invariants

- **`@torch.compile` constraints.** These helpers run inside the compiled DC solve, so they must obey: no in-place tensor writes; no Python-side scalar branches on tensor values; no autograd reads of attributes that change across calls. Respecting them keeps the whole solve a single fused kernel.
- **Boundary slots are ignored, not validated.** In `solve_tridiagonal` the `sub` entry at index 0 and the `sup` entry at the last index are unused boundary slots; in `solve_block_tridiagonal` the `sub` block at $k=0$ and the `sup` block at $k=N-1$ are unused. The algorithms simply never read them — callers need not zero them for the Thomas path.
- **Block-count semantics ($B$).** `solve_block_tridiagonal` works for any block size $B$ from the same code path: $B=1$ reduces to scalar Thomas and matches `solve_tridiagonal` numerically up to fp64 round-off, and $B=2$ is the coupled two-rail wire Newton the 1T1R nested solver uses; larger $B$ runs through the identical sweep for any topology that couples more node classes per row. The single-block fast path ($N=1$) is one $B \times B$ `torch.linalg.solve`.
- **Wire-ladder layout for the residual builders.** `col_wire_kcl_residual` and `col_driver_current` assume the wire runs along the row axis with the driver at the boundary. `segment_g` (shape `(row_num,)`) encodes the per-segment conductances along that wire; its ordering is boundary-dependent, with the driver segment at the boundary end and the cell-to-cell segments following toward the far end.
- **Sign / unit conventions.** The KCL residual is returned in uA with the convention $r = i_{\mathrm{inject}} + \Delta v_{\mathrm{left}} \cdot g_{\mathrm{left}} + \Delta v_{\mathrm{right}} \cdot g_{\mathrm{right}}$, where $\Delta v_{\mathrm{left}}[k] = v[k] - v[k-1]$ (and $v[0] - v_{\mathrm{drive}}$ at the driver end) and $\Delta v_{\mathrm{right}}[k] = v[k] - v[k+1]$ (zero at the far end, which has no right segment). Node voltages are V, segment conductances uS, currents uA.
- **Elementwise derivative diagonal.** `elementwise_diff(fn, *, wrt, **kwargs)` returns the per-element derivative of an elementwise device function via one autograd backward pass. Because `fn` is elementwise, the derivative of the summed output with respect to each input element is that element's own derivative, so a single backward yields the full Jacobian diagonal without ever materialising off-diagonal terms.

## Performance & resources

- **Thomas vs block-Thomas cost.** Scalar Thomas is $O(N)$ elementwise work per system; block-Thomas is $O(N \cdot B^3)$ with one $B \times B$ solve per row, so memory stays $O(N \cdot B^2)$ per leading-batch instance. The block solver carries one extra trailing rank versus the scalar solver even at $B=1$.
- **Unrolled-loop fusion.** Because the sweeps build Python lists and `torch.stack` once, the per-step ops are pure-functional and Inductor fuses the unrolled forward and back sweeps; there is no per-iteration kernel launch in the compiled graph.
- **No materialised dense matrix.** The Thomas path never assembles the full $N B \times N B$ system, so its per-instance memory does not scale with $N^2$.

## Gotchas

- **`solve_block_tridiagonal` has no `dim` argument by design.** Passing data in a different axis order silently mis-indexes the block sweep rather than erroring; transpose to `[..., N, B, B]` / `[..., N, B]` before calling.
- **`segment_g` ordering is boundary-dependent — verify it matches the wire topology.** A mismatched ordering shifts every wire conductance and quietly corrupts the residual rather than erroring; `col_wire_kcl_residual` and `col_driver_current` share the same convention, so an inconsistency corrupts both.
- **`B=1` block-Thomas is not bit-identical to scalar Thomas.** It agrees only up to fp64 round-off because the $1 \times 1$ `torch.linalg.solve` path differs arithmetically from the scalar division in `solve_tridiagonal`; use the scalar helper when exact reproducibility against the scalar path matters.

## Known limitations

- **No pivoting in the Thomas sweeps.** Both `solve_tridiagonal` and `solve_block_tridiagonal` assume the modified diagonal (block) stays non-singular through the forward sweep; they perform no row/block pivoting. Well-posedness is the responsibility of the topology's Newton formulation, not these primitives.
- **Builders cover the column-oriented wire only.** `col_wire_kcl_residual` / `col_driver_current` are specialised to the `dim=-1` column wire; a row-oriented wire uses the sibling row-wire builders in the same module rather than a `dim` parameter on these.

---

- **Reference**: N/A — these are software primitives; the consuming numerical method is specified by the owning topology.
- **Implementation**: `neurox/xbar/solver.py`
- **Tests**: `tests/test_solve_tridiagonal.py`, `tests/test_block_tridiagonal.py`, `tests/test_block_tridiagonal_dense.py`, `tests/test_block_tridiagonal_pcr.py`
- **Decisions**: N/A — no ADR governs this module.

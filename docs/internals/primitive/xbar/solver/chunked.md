# Chunked solver

Chunking sits **between** the array and the solver. `ChunkedSolver(solver, chunk_size=...)` wraps a `Solver` and exposes `solve_dc(*, leading, measure, measure_tensors, **kwargs)` — the wrapped solver's own call plus the leading and the per-chunk measurement — so an array works at its original shapes and holds no chunking logic, while the solver faithfully solves one fixed shape. The leading-axis machinery the wrapper runs on is the [chunking layer](chunking.md).

## Design decisions

- **Argument handling is signature-agnostic.** An argument that is a snap gets chunk-sliced; every other argument passes through untouched — modules, protocol objects, and scalar circuit constants that carry no leading, such as the two rail-link resistances. The wrapper therefore encodes nothing about the wrapped signature.
- **Every chunk is folded; none is retained.** `measure` is called on each chunk with that chunk's DCOP plus every sliced operand, and returns a frozen dataclass of the small tensors that outlive the chunk — the port state and the caller's per-instance billing. Only those are reassembled, so a `SolverDcop` never leaves the loop and the grid-shaped node voltages die with the chunk that produced them. The alternative, returning the full-leading DCOP, costs six grid-shaped fields at the whole leading: hundreds of gigabytes at a serialized macro's leading, against kilobytes for the port state.
- **The caller declares the survivors; the layer never infers them.** The measurement is the caller's, so what survives a chunk is a modelling decision, not a rank heuristic — the same principle that has the caller declare the leading. `measure_tensors` reach `measure` alone: the wrapped solver must never receive a keyword it does not declare, so a boundary the solve does not read (the word-line drive, a scheme's per-column input drive) still gets sliced by the same generic rule. They travel as bare tensors — the slicing rule is a tensor's, and a dataclass is a walk over tensors rather than a precondition for slicing one, so nothing has to be wrapped merely to be seen.
- **The eager island is the chunk loop.** `ChunkedSolver.solve_dc` carries `@torch.compiler.disable(recursive=False)`. Its trip count `ceil(leading / chunk_size)` is a runtime value that would unroll or specialize an enclosing compiled graph, while the per-chunk `solver.solve_dc` inside it stays the compiled leaf at one fixed shape. See [compile/scheme_a_regional](../../../compile/scheme_a_regional.md).
- **Snaps are sampled once, at full shape.** The caller snapshots before reaching this layer, so every per-call noise draw happens once over the whole call and the layer only re-views the result. `chunk_size` is therefore a memory knob and never a numerical one — the returned measurement is bit-identical across chunk sizes. No snapshot method anywhere carries a chunk-selection argument; slicing acts on the snap object afterwards.
- **The array states the leading; nothing infers it.** The array already computes the call's broadcast leading with plain slicing plus `torch.broadcast_shapes` — the trailing rank of each operand is a property only that operand's owner knows — so it passes that value down and `trailing = shape[len(leading):]` follows for every field. Inferring instead would need a declared trailing rank per snap — `[col, row]` and `[batch, col]` are both rank 2 and mean different things — which would put a crossbar-tiling concept on the analog layer and on the `ClampSnap` role.

## Contracts & invariants

- **Every sliced operand arrives at the call's full leading.** **Every** snap tensor field carries it, the 0-d constants included; `ChunkedSolver.solve_dc` checks the invariant once per call, so a field that is short of it fails before any chunk runs rather than having its own trailing block gathered.

---

- **Reference**: N/A — a memory-tiling layer with no physics twin
- **Implementation**: `neurox/primitive/xbar/solver/chunked.py`
- **Tests**: `tests/primitive/xbar/test_chunked_solver.py`

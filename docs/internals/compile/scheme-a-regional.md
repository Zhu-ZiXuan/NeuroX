# Scheme A — Regional Compilation

**Status: Active.** The current production scheme. Minimal change that lets the physical-xbar `matmul` compile without the compile graph exploding.

## The problem it solves

The 1T1R physical xbar runs a DC circuit solve per VMM, and that solve is the only place in the library where a naive compile boundary fails. Two properties of the solve collide with `torch.compile`:

- **The solve is heavy and deeply nested.** `CircuitCore1T1R.cim_read` settles the array to its DC operating point through `NestedSolver1T1R.solve_dc`, which runs a fixed number of outer/inner Newton iterations; each iteration assembles cell conductances, wire KCL residuals, and a coupled block-tridiagonal system. One `solve_dc` is already a large graph.
- **The solve is run in a memory-bounding chunk loop.** `cim_read`'s leading batch is very large (im2col positions × batch × slice × column). Solving it in one shot materializes every instance's node voltages at once and runs out of memory, so `cim_read` walks the leading in chunks, releasing each chunk's operating point before the next allocates. Chunking is load-bearing and cannot be dropped to make compilation easier.

If the macro `matmul` graph traced straight through `cim_read`, the chunk loop would be unrolled into the main graph, so the graph would hold roughly `chunk_count` copies of the whole solve. Worse, `chunk_count = ceil(leading / chunk_size)` is a runtime value, so the loop is not a static `range` the graph can hold — tracing it would either specialize on a concrete count (recompiling per leading) or unroll an unbounded region. Either way both compile time and graph size blow up while the chunk loop, the very thing that bounds memory, is the thing that does not belong in the graph.

The historical note that this blow-up came from the SAR ADC bit-loop is stale; the SAR loop is short and the cost is the chunked solver.

## The PyTorch mechanism it leans on

> Mechanism behaviour below is observed on the project's pinned PyTorch and **may change across versions** — re-confirm against the installed version before relying on an edge.

- **Regional compilation** ([recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)): compiling a small repeated region and reusing it from eager Python control flow, instead of compiling one large graph. The compile cost is set by the region's graph, not by how many times the loop runs.
- **`torch.compiler.disable(recursive=False, reason=...)`** ([docs](https://docs.pytorch.org/docs/stable/compile/programming_model.compiler_disable.html)): marks a function as an eager island so the outer graph breaks at it rather than tracing in. `recursive=False` is the documented form that still allows nested `@torch.compile` callees to compile. The recursive semantics are an edge with conflicting reports ([#123771](https://github.com/pytorch/pytorch/issues/123771) — inner `@torch.compile` may compile even under the default `recursive=True`; [#148787](https://github.com/pytorch/pytorch/issues/148787) — `recursive=False` mutating the wrapped function); the explicit `recursive=False` is chosen for being the documented-correct, intent-pinning form, and the inner-leaf-compiles behaviour must be re-confirmed per version.
- **`@torch.compile(dynamic=False)`** on the leaf: the chunk shape is constant, so a static-shape graph is built once and reused; cross-instance reuse rides on dynamo lifting nn.Module buffers as shape-guarded graph inputs rather than identity-guarded constants.

`torch.compiler.allow_in_graph` is **not** usable here: it only stops the Dynamo front-end from tracing in; the Inductor back-end still traces the body, so it does not hide the solver's complexity. The true opaque-node mechanism is a custom op ([scheme C](scheme-c-custom-op.md)).

## Implementation form

```text
macro.matmul        @torch.compile(dynamic=True)          # user entry; batch shapes vary
  -> vec_mat_mul    (compiled-path)                       # index / readout math
       -> cim_read  @torch.compiler.disable(recursive=False)   # eager island; owns the chunk loop
            for chunk in iter_chunks(...):                # Python loop, never in a graph
                solve_dc(chunk)                           # the compiled leaf
            reassemble; log energy/latency (eager)
       -> readout.readout(...)                            # compiled-path
  -> digital aggregate                                    # compiled-path
solver.solve_dc     @torch.compile(dynamic=False)         # fixed chunk shape -> one shared graph
```

The block-tridiagonal kernel inside the solve uses the **Thomas sweep** (`solve_block_tridiagonal`). Thomas has a deep unrolled graph (linear in the row count), so its first compile is long — on the order of ~10 min for the 28nm tile. That cost is paid **once**: with a uniform `solve_chunk_size` the leaf has a single shape, so the one long compile is shared across every chunk, VMM, and macro layer, and the FX graph cache carries it across processes. The payoff is decisive at runtime — compiled-Thomas measured fastest **and** leanest of every backend, well ahead of the log-depth `solve_block_tridiagonal_pcr` (which compiles far quicker but runs several times slower and heavier because of its `O(N \log N)` block-matrix work). PCR / dense stay as numerically interchangeable references, but Thomas is the compiled hot path.

### Boundary map (authoritative)

Every deviation from the compiled-path default lives here; module docs only point back:

- **Eager island** — `CircuitCore1T1R.cim_read` (`@torch.compiler.disable(recursive=False)`). Owns the chunk loop, list accumulation, snapshot indexing, and the profiler emit. The [contracts](contracts.md) are lifted inside it.
- **Regional leaf** — `Solver1T1R.solve_dc` on every concrete solver (`@torch.compile(dynamic=False)`). Fixed chunk shape; obeys the contracts.
- **Disabled hooks** — `ProfileMixin._log_dynamic_energy` / `_log_latency` (`@torch.compiler.disable`); side-channel writes, placed after the kernel math so fusion is unaffected.
- **Compiled entry** — `XbarMacro.matmul` on every concrete macro (`@torch.compile(dynamic=True)`).

## Trade-offs

- **vs. tracing the solve into the macro graph (the failing baseline).** Regional compilation gives a compile cost set by one solve's graph, independent of `chunk_count` and of the leading batch, and keeps chunking's memory bound. It loses whole-program fusion: the macro graph breaks at `cim_read`, so math on either side of the xbar read is not fused across the read.
- **vs. leaving the solve eager (no leaf decoration).** Compiling the leaf lets Inductor fuse the dense per-cell / per-wire kernels of the solve, which dominate VMM cost. The price is that the leaf is a separate compiled artifact whose shape stability must be managed.
- **vs. removing chunking to allow one graph.** Rejected: chunking is the memory bound, not a compile convenience; the leading batch does not fit unchunked.
- **Boundary placement: `cim_read` vs higher.** The island is drawn at `cim_read` (not at `vec_mat_mul`) so that `vec_mat_mul`'s index and readout math stay on the compiled path and only the chunk loop is excluded. Drawing it higher would needlessly exclude readout from compilation.
- **`dynamic=True` at the entry, `dynamic=False` at the leaf.** The entry must absorb user batch shapes; the leaf must pin the chunk shape to get a single reusable graph. The split is deliberate.
- **`fullgraph=False`.** The eager island and profiler hooks break by design. A consumer that needs `fullgraph=True` is not served by this scheme — that is [scheme C](scheme-c-custom-op.md).

## Performance and resources (theoretical)

- **Compile cost** scales with the size of one `solve_dc` graph, and is paid **once** per distinct compile signature (see [contracts](contracts.md) recompile triggers), not per chunk and not per leading batch. Thomas's graph is deep (linear in the row count), so that one compile is long (~10 min); the uniform chunk shape keeps it to a single signature, and the on-disk cache removes it on subsequent runs.
- **Peak memory** is bounded by the chunk working set, independent of total leading — the eager loop releases each chunk before the next allocates. But the *compiled* chunk working set is larger than the eager one: Inductor co-allocates a graph's intermediates rather than freeing them step-by-step the way eager Python does, so the compiled solve's per-chunk peak exceeds the eager solve's at the same chunk size. Chunk sizes therefore have to be tuned for the compiled path, not the eager one.
- **Cross-instance reuse**: when the leaf's compile signature is shape-only, every macro layer of the same geometry reuses one graph; the on-disk graph cache carries it across processes. Whether the signature stays shape-only depends on the implicit buffer-lifting mechanism — the risk this introduces is what [scheme B](scheme-b-deobjectified.md) removes.

## Risks and failure modes

- **The eager island suppressing the leaf compile.** If a PyTorch version makes the disabled `cim_read` propagate "do not compile" into its callees, `solve_dc` would silently run eager and the bottleneck would never compile — no error, just slow. This is the recursive-disable edge above; it must be re-confirmed (e.g. that the leaf produces exactly one compiled graph) whenever the PyTorch version changes.
- **Per-layer recompilation.** If the leaf's compile signature accidentally keys on object identity (the solver, an RRAM/NMOS buffer) rather than tensor shape, every layer cold-compiles instead of sharing one graph ([#141589](https://github.com/pytorch/pytorch/issues/141589) tracks this class of recompile). The symptom is many cold compiles at model build; the fix is [scheme B](scheme-b-deobjectified.md).
- **Remainder chunk.** A leading not divisible by the chunk size yields one smaller trailing chunk, a second shape, hence one extra leaf graph. Acceptable; pad-to-full-chunk would collapse it to one graph at the cost of wasted solve work.
- **Snapshot / dataclass arguments.** The leaf takes device-snapshot dataclasses; if their PyTree structure or a field's Python type drifts call-to-call, the leaf recompiles. Keep snapshot structure stable.
- **Shape drift.** Changing `row_num`, `physical_col_num`, chunk size, dtype, device, or solver iteration counts is a new compile signature — expected, but worth knowing when a workload suddenly recompiles.
- **Wrong block-tridiagonal backend.** Swapping the compiled leaf to PCR (or dense) to shorten the cold compile is a trap: it runs several times slower and heavier than compiled-Thomas. The long Thomas compile is a one-time, cached, single-signature cost; the runtime regression of the log-depth backends is paid on every solve.
- **Non-uniform chunk shapes multiply the long compile.** Thomas's compile is long, so if the leaf sees more than one chunk shape (a remainder chunk, or per-layer inst sizes left un-chunked) each extra shape pays it again. Keep `solve_chunk_size` set so every chunk is one uniform shape.
- **Compile cost on shape-sweeping callers.** Tests and calibration tools that sweep shapes pay one cold compile per shape. Run them with the graph cache enabled, or with compilation disabled, when only eager logic is under test.
- **Compiled peak memory exceeds eager — re-tune the chunk size.** Because Inductor co-allocates the solve's intermediates (the block-tridiagonal stages are all resident at once, unlike the eager solver's step-by-step release), a `solve_chunk_size` tuned for the eager solver can OOM under compilation. `solve_chunk_size` is the per-chunk leading directly (the peak-memory budget), so for the compiled path it must be set smaller than an eager run would tolerate — a transformer FFN at the eager-tuned size overran GPU memory. The readout sits on the compiled macro path (not in the eager island), so its intermediates are co-allocated by Inductor and add to the compiled peak. Treat `solve_chunk_size` as compiled-path memory tuning, not eager-path tuning.

## Open questions

- No `fullgraph=True` path — the eager island breaks the graph by design.
- Cross-instance sharing rides on an implicit, version-dependent mechanism rather than a guaranteed shape-only cache key; [scheme B](scheme-b-deobjectified.md) is the hardening.

## See also

- [contracts](contracts.md), [scheme B](scheme-b-deobjectified.md), [scheme C](scheme-c-custom-op.md)
- Implementation: `neurox/macro/xbar/*.py`, `neurox/xbar/_1t1r/{offset,circuit_core,nested_solver,full_jacobian_solver}.py`, `neurox/xbar/solver.py`, `neurox/common/mixin/profile.py`

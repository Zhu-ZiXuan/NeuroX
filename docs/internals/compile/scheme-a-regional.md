# Scheme A — Regional Compilation

**Status: Active.** The current production scheme. The library self-compiles exactly one region — the DC-solver leaf — and leaves the rest of the macro forward eager.

## What compiles, and what does not

The library does **not** self-compile the macro forward. `XbarMacro.matmul`, `Offset1T1RXbar.vec_mat_mul`, the readout chain, and the digital aggregation all run eager. They are written to be compile-*friendly* (they obey the [contracts](contracts.md)), so a caller may `torch.compile` an entire model, but the library does not force it — consistent with the project rule that compilation is the caller's policy. The one region the library compiles itself is the **DC-solver leaf** `Solver1T1R.solve_dc` (`@torch.compile(dynamic=False)`), reached through `CircuitCore1T1R.cim_read`'s eager island.

## The problem it solves

The DC solve is the one place a self-compiled boundary is both necessary and hard.

- **The solve is heavy and deeply nested.** `cim_read` settles the array to its DC operating point through `solve_dc`, which runs a fixed number of outer/inner Newton iterations; each iteration assembles cell conductances, wire KCL residuals, and a coupled block-tridiagonal system. One `solve_dc` is already a large graph, and its dense per-cell / per-wire kernels are exactly what Inductor fuses — this is where the compile win is.
- **The solve runs in a memory-bounding chunk loop.** `cim_read`'s leading batch is very large (im2col positions × batch × slice × column). Solving it in one shot materializes every instance's node voltages at once and runs out of memory, so `cim_read` walks the leading in chunks, releasing each chunk's operating point before the next allocates. Chunking is load-bearing. Its trip count `ceil(leading / solve_chunk_size)` is a runtime value, so the loop is not a static `range` a graph can hold — tracing it would either specialize on a concrete count or unroll an unbounded region.
- **The surrounding forward is shape-diverse and profiler-instrumented.** The macro forward's leading shape *and rank* vary per layer (a classifier activation carries fewer leading dims than an encoder one), it carries many size-1 tile dims (`M`, `Sw`, `Tc`, `Tr`) that dynamo specializes, and every analog leaf logs through a `@torch.compiler.disable` profiler hook. Self-compiling that forward shatters it into ~20 profiler-/island-separated frames, each of which recompiles on every distinct tile geometry and input rank — `dynamic=True` absorbs batch *size* but cannot absorb a size-1↔N flip or a rank change — so the per-frame graph count blows past dynamo's `cache_size_limit` and the forward evicts to eager, after each frame has already paid a slow Inductor compile (the readout/ADC chain is the worst).

So the boundary is drawn tightly around the solve. `cim_read` is an eager island that owns the chunk loop, and `solve_dc` inside it is the one compiled leaf. The leaf is **shape-stable**: `cim_read`'s per-chunk advanced-indexing flattens the broadcast leading to a fixed `(chunk_size, 1, row)`, so the solve graph has a single shape regardless of which layer called it — the per-layer diversity that defeats whole-forward compilation never reaches the leaf. The readout / ADC chain, being both shape-diverse and slow to compile for only light-op fusion, is deliberately left eager.

## The PyTorch mechanism it leans on

> Mechanism behaviour below is observed on the project's pinned PyTorch and **may change across versions** — re-confirm against the installed version before relying on an edge.

- **Regional compilation** ([recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html)): compiling a small repeated region and reusing it from eager Python control flow, instead of compiling one large graph. The compile cost is set by the region's graph, not by how many times the loop runs.
- **`torch.compiler.disable(recursive=False, reason=...)`** ([docs](https://docs.pytorch.org/docs/stable/compile/programming_model.compiler_disable.html)): marks a function as an eager island so an enclosing graph breaks at it rather than tracing in. `recursive=False` is the documented form that still allows nested `@torch.compile` callees to compile. The recursive semantics are an edge with conflicting reports ([#123771](https://github.com/pytorch/pytorch/issues/123771) — inner `@torch.compile` may compile even under the default `recursive=True`; [#148787](https://github.com/pytorch/pytorch/issues/148787) — `recursive=False` mutating the wrapped function); the explicit `recursive=False` is chosen for being the documented-correct, intent-pinning form, and the inner-leaf-compiles behaviour must be re-confirmed per version.
- **`@torch.compile(dynamic=False)`** on the leaf: the chunk shape is constant, so a static-shape graph is built once and reused; cross-instance reuse rides on dynamo lifting nn.Module buffers as shape-guarded graph inputs rather than identity-guarded constants.

`torch.compiler.allow_in_graph` is **not** usable here: it only stops the Dynamo front-end from tracing in; the Inductor back-end still traces the body, so it does not hide the solver's complexity. The true opaque-node mechanism is a custom op ([scheme C](scheme-c-custom-op.md)).

## Implementation form

```text
macro.matmul        (eager; compile-friendly — a caller may torch.compile the model)
  -> vec_mat_mul    (eager)                               # index / readout math
       -> cim_read  @torch.compiler.disable(recursive=False)   # eager island; owns the chunk loop
            for chunk in iter_chunks(...):                # Python loop, never in a graph
                solve_dc(chunk)                           # the compiled leaf  <- the only self-compiled region
            reassemble; log energy/latency (eager)
       -> readout.readout(...)                            # eager
  -> digital aggregate                                    # eager
solver.solve_dc     @torch.compile(dynamic=False)         # fixed chunk shape -> one shared graph
```

The block-tridiagonal kernel inside the solve uses the **Thomas sweep** (`solve_block_tridiagonal`). Thomas has a deep unrolled graph (linear in the row count), so its first compile is long — on the order of ~10 min for the 28nm tile. That cost is paid **once**: with a uniform `solve_chunk_size` the leaf has a single shape, so the one long compile is shared across every chunk, VMM, and macro layer, and the FX graph cache carries it across processes. The payoff is decisive at runtime — compiled-Thomas measured fastest **and** leanest of every backend, well ahead of the log-depth `solve_block_tridiagonal_pcr` (which compiles far quicker but runs several times slower and heavier because of its $O(N \log N)$ block-matrix work). PCR / dense stay as numerically interchangeable references, but Thomas is the compiled hot path.

### Boundary map (authoritative)

Every deviation from the eager default lives here; module docs only point back:

- **Eager island** — `CircuitCore1T1R.cim_read` (`@torch.compiler.disable(recursive=False)`). Owns the chunk loop, list accumulation, snap indexing, and the profiler emit. Under a caller-applied compile it stays an eager island while still letting the nested leaf compile.
- **Regional leaf** — `Solver1T1R.solve_dc` on every concrete solver (`@torch.compile(dynamic=False)`). The **only** region the library self-compiles. Shape-stable: the chunk indexing in `cim_read` flattens every call to one `(chunk_size, 1, row)` shape. Obeys the [contracts](contracts.md).
- **Disabled hooks** — `ProfileMixin._log_dynamic_energy` / `_log_latency` (`@torch.compiler.disable`); side-channel writes, placed after the kernel math so fusion is unaffected.
- **No self-compiled forward** — `XbarMacro.matmul`, `vec_mat_mul`, readout, and digital aggregation run eager. They obey the contracts so a caller *may* compile them, but the library does not self-decorate them.

## Trade-offs

- **vs. self-compiling the whole macro forward (the prior approach).** Rejected. The forward's per-layer geometry and input-rank diversity, plus the profiler-hook graph breaks, shatter it into roughly `frames × geometries × ranks` graphs that evict to eager at `cache_size_limit` — and each frame still pays a slow readout/ADC Inductor compile — all to fuse light analog/digital pointwise ops. The solver leaf alone is shape-stable and compiles once.
- **vs. leaving the solve eager (no leaf decoration).** Compiling the leaf lets Inductor fuse the dense per-cell / per-wire kernels of the solve, which dominate VMM cost. The price is a separate compiled artifact whose shape stability must be managed — but the chunk-flattening guarantees it.
- **vs. removing chunking to allow one graph.** Rejected: chunking is the memory bound, not a compile convenience; the leading batch does not fit unchunked.
- **vs. raising `cache_size_limit` and self-compiling the forward anyway.** Even with the limit lifted, each geometry/rank still cold-compiles the slow readout/ADC chain and the count grows with model depth; a caller whose workload justifies whole-model fusion can opt in and tune the limit themselves.
- **`dynamic=False` at the leaf.** The leaf must pin the chunk shape to get a single reusable graph; the chunk-flattening makes that shape layer-independent.

## Performance and resources (theoretical)

- **Compile cost** is the size of one `solve_dc` graph — the only self-compiled region — paid **once** per distinct compile signature (see [contracts](contracts.md) recompile triggers), not per chunk and not per leading batch. Thomas's graph is deep (linear in the row count), so that one compile is long (~10 min); the uniform chunk shape keeps it to a single signature, and the on-disk cache removes it on subsequent runs.
- **Peak memory** is bounded by the chunk working set, independent of total leading — the eager loop releases each chunk before the next allocates. But the *compiled* chunk working set is larger than the eager one: Inductor co-allocates a graph's intermediates rather than freeing them step-by-step the way eager Python does, so the compiled solve's per-chunk peak exceeds the eager solve's at the same chunk size. Chunk sizes therefore have to be tuned for the compiled path, not the eager one.
- **Cross-instance reuse**: when the leaf's compile signature is shape-only, every macro layer reuses one graph; the on-disk graph cache carries it across processes. Whether the signature stays shape-only depends on the implicit buffer-lifting mechanism — the risk this introduces is what [scheme B](scheme-b-deobjectified.md) removes.

## Risks and failure modes

- **The eager island suppressing the leaf compile.** If a PyTorch version makes the disabled `cim_read` propagate "do not compile" into its callees, `solve_dc` would silently run eager and the bottleneck would never compile — no error, just slow. This is the recursive-disable edge above; it must be re-confirmed (e.g. that the leaf produces exactly one compiled graph) whenever the PyTorch version changes.
- **Per-layer recompilation of the leaf.** The leaf is shape-stable, so this can only happen if its compile signature accidentally keys on object identity (the solver, an RRAM/NMOS buffer) rather than tensor shape, cold-compiling per layer instead of sharing one graph ([#141589](https://github.com/pytorch/pytorch/issues/141589) tracks this class of recompile). The symptom is many cold compiles at model build; the fix is [scheme B](scheme-b-deobjectified.md).
- **Caller-applied whole-model compile re-exposes the forward's diversity.** If a caller wraps the model in `torch.compile`, the macro forward's size-1 / rank specialization and profiler-hook graph breaks return as the *caller's* tuning problem — raise `cache_size_limit`, or normalize the activation rank before the macro. The library deliberately does not self-compile the forward, so the default path never pays this.
- **Remainder chunk.** A leading not divisible by the chunk size yields one smaller trailing chunk, a second shape, hence one extra leaf graph. Acceptable; pad-to-full-chunk would collapse it to one graph at the cost of wasted solve work.
- **Snap / dataclass arguments.** The leaf takes device-snap dataclasses; if their PyTree structure or a field's Python type drifts call-to-call, the leaf recompiles. Keep snap structure stable.
- **Shape drift.** Changing `row_num`, `physical_col_num`, chunk size, dtype, device, or solver iteration counts is a new compile signature — expected, but worth knowing when a workload suddenly recompiles.
- **Wrong block-tridiagonal backend.** Swapping the compiled leaf to PCR (or dense) to shorten the cold compile is a trap: it runs several times slower and heavier than compiled-Thomas. The long Thomas compile is a one-time, cached, single-signature cost; the runtime regression of the log-depth backends is paid on every solve.
- **Non-uniform chunk shapes multiply the long compile.** Thomas's compile is long, so if the leaf sees more than one chunk shape (a remainder chunk, or per-layer inst sizes left un-chunked) each extra shape pays it again. Keep `solve_chunk_size` set so every chunk is one uniform shape.
- **Compile cost on shape-sweeping callers.** Tests and calibration tools that sweep shapes pay one cold compile per shape. Run them with the graph cache enabled, or with compilation disabled, when only eager logic is under test.
- **Compiled peak memory exceeds eager — re-tune the chunk size.** Because Inductor co-allocates the solve's intermediates (the block-tridiagonal stages are all resident at once, unlike the eager solver's step-by-step release), a `solve_chunk_size` tuned for the eager solver can OOM under compilation. `solve_chunk_size` is the per-chunk leading directly (the peak-memory budget), so for the compiled path it must be set smaller than an eager run would tolerate — a transformer FFN at the eager-tuned size overran GPU memory. Treat `solve_chunk_size` as compiled-path memory tuning, not eager-path tuning.

## Open questions

- No `fullgraph=True` path through the macro — the eager island and profiler hooks break the graph by design; [scheme C](scheme-c-custom-op.md) is the fullgraph route.
- Cross-instance sharing rides on an implicit, version-dependent mechanism rather than a guaranteed shape-only cache key; [scheme B](scheme-b-deobjectified.md) is the hardening.

## See also

- [contracts](contracts.md), [scheme B](scheme-b-deobjectified.md), [scheme C](scheme-c-custom-op.md)
- Implementation: `neurox/macro/xbar/*.py`, `neurox/xbar/_1t1r/{offset,circuit_core,nested_solver}.py`, `neurox/xbar/solver.py`, `neurox/common/mixin/profile.py`

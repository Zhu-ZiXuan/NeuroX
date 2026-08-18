# Compile boundary

The library self-compiles exactly one region, the DC-solver leaf, and runs everything above it eagerly. The eager part is written to stay traceable, so a caller may wrap a whole model in `torch.compile`, but the library never makes that choice on the caller's behalf. Maintaining any function on this path means knowing which side of the boundary it sits on, because the two sides admit different code: the eager side may loop on a runtime trip count, free memory step by step, and read host state, and the compiled side may not. The invariants the traceable side obeys are in [code_style](../conventions/code_style.md); this page maps the boundary itself and the failure modes that cross it.

## Where the boundary sits

- **Eager island** — `ChunkedSolver.solve_dc`, the only chunk loop in the library; every array reaches a solve through it and none holds a loop of its own. It carries `@torch.compiler.disable(recursive=False)`, so it stays eager even inside a caller's compiled model while the compiled leaf nested below it still compiles. It is the one place a runtime trip count, per-chunk allocation release, and host-side accumulation are legal.
- **Compiled leaf** — `ColBlColSlSolver._solve_dc_impl` under `@torch.compile(dynamic=False)`, the only region the library compiles for itself. Its `solve_dc` entry point stays an undecorated wrapper so the prober gate is read outside the graph.
- **Recorder gates** — the `RecorderBase` classmethods every side channel passes through (`current`, `active`, `submit`, `_submit_record`), each `@torch.compiler.disable`, with an undecorated `submit` override rejected at class definition. The active slot is host state no guard watches: traced, an empty slot folds into the graph as a constant and that recorder family collects nothing ever after. Every emit site therefore breaks the graph at its gate read, and again at the hand-over.
- **Profile consumers** — the `ProfileMixin` hooks an emitter calls, `_is_dynamic_energy_profile_active` and `_record_dynamic_energy`. They are ordinary traceable methods that reach the disabled gates, so the energy tensor an emitter builds stays in its caller's graph and only the query and the hand-over leave it. For the boundary this means a billing site carries a graph break wherever it is placed.

Nothing else is decorated. The unit forward, the engine `matmul`, the macro `vec_mat_mul`, the readout chain, and the digital aggregation are compile-friendly, not compiled.

## Why the boundary is drawn around the solve

Three properties of the solve decide it, none of them a property of the forward.

The solve is where compilation pays. It settles the array through nested outer and inner Newton iterations, each assembling cell conductances, wire KCL residuals, and a coupled block-tridiagonal system — dense elementwise work that fuses well and dominates the runtime of an analog matmul.

The chunk loop cannot live inside a graph. Its trip count follows from the call's leading extent and the chunk size, both runtime values, so tracing it would either specialize the graph on one count or unroll an unbounded region. Chunking is not optional: the leading is the product of the caller's batch axes and the macro's own placement axes, and solving it in one block materializes every instance's Newton state at once.

The leaf is the only shape-stable point in the stack. Chunk slicing flattens whatever the caller's leading was into one `(chunk_size, col, row)` cell grid, so every layer enters the leaf at the same shape. Above the leaf, the leading shape and rank vary per layer, placement axes are size-1 under some mappings and not others, and every billing site breaks the graph; self-compiling that forward yields many small frames, each re-keyed by geometry that `dynamic=True` cannot absorb, since a size-1-to-N flip or a rank change is not a dynamic dimension. The single leaf shape is also what makes the cost affordable: the block-tridiagonal sweep unrolls linearly in the row count, so its graph is deep and its cold compile long, but one compile signature means one graph shared by every chunk, every VMM, and every macro layer, cached on disk across processes.

## Recompile keys are wider than shape

`dynamic=False` pins the leaf to a static shape, but shape is not the whole cache key: the Python type of every argument enters it, and for a scalar argument so does the value. Exceeding the recompile budget for one code object does not raise — the leaf falls back to eager and the run is merely slow.

- The two rail resistances arrive as Python floats, so each distinct value compiles its own graph. Sweeping resistance or geometry under compilation is what exhausts the budget; calibration tools and the test suite therefore run under `TORCH_COMPILE_DISABLE=1`.
- Snaps arrive as frozen dataclasses, so their PyTree structure is part of the key. An optional field present on one call and absent on the next re-keys the leaf.
- The cell model and both clamp drivers arrive as module arguments. Whether every macro layer shares one graph or each cold-compiles depends on their state being lifted as guarded graph inputs rather than pinned per instance; the symptom of the latter is a burst of cold compiles at model build instead of one.
- A `solve_chunk_size` of zero hands the whole leading to the leaf unpadded, so the leaf then sees one shape per call site rather than one shape overall.

## Compiled peak memory exceeds eager

Chunking bounds the peak of the whole call, which makes `solve_chunk_size` the single knob over it — and the knob has to be tuned against the compiled path. Inductor co-allocates a graph's intermediates where eager Python frees them as it goes, so the compiled per-chunk peak exceeds the eager one at the same chunk size, and a value proven in an eager run can run out of memory once the leaf compiles.

## Tail chunks are padded

A short final chunk is padded up to `solve_chunk_size` by repeating its last leading coordinate, which is how every chunk reaches the leaf at one shape. A padded position is a duplicate of a real one, and duplicates are dropped only when the fold writes the chunk back to its global indices. A per-chunk measurement therefore keeps the chunk axis intact and reduces only over the axes behind it; a reduction across the chunk axis inside the measurement would count the duplicated positions, which is how a padded tail turns into inflated energy or an inflated aggregate rather than into a shape error.

# Compile boundary

The library self-compiles the numerical work repeated for every array chunk and runs the chunk orchestration and everything above it eagerly. The eager part is written to stay traceable, so a caller may wrap a whole model in `torch.compile`, but the library never makes that choice on the caller's behalf. Maintaining any function on this path means knowing which side of the boundary it sits on, because the two sides admit different code: the eager side may loop on a runtime trip count, free memory step by step, and read host state, and the compiled side may not. The invariants the traceable side obeys are in [code_style](../conventions/code_style.md); this page maps the boundary itself and the failure modes that cross it.

## Where the boundary sits

- **Eager island** — `execute_chunked`, the only chunk loop in the library; every chunked array reaches a solve through it and none holds a loop of its own. It carries `@torch_compiler_disable(recursive=False)`, so it stays eager even inside a caller's compiled model while the injected numerical leaves still compile. It is the one place a runtime trip count, per-chunk allocation release, and host-side accumulation are legal.
- **Compiled solver leaf** — `_solve_col_bl_col_sl_dc_impl` under `@torch.compile(dynamic=False)`. Its `solve_col_bl_col_sl_dc` entry point stays undecorated so the prober gate is read outside the graph.
- **Compiled projection leaf** — the array's final `_project_dcop` entry point under `@torch.compile(dynamic=False)`. It dispatches to the overridable `_project_dcop_impl`, so the base port projection, optional capacitive-energy calculation, and topology-specific reductions follow one compiled fixed-shape path without requiring every concrete array to repeat the decorator. The profiler gate may split that path as described below.
- **Recorder gates** — the `RecorderBase` classmethods every side channel passes through (`current`, `active`, `submit`, `_submit_record`), each `@torch_compiler_disable`, with an undecorated `submit` override rejected at class definition. The active slot is host state no guard watches: traced, an empty slot folds into the graph as a constant and that recorder family collects nothing ever after. Every emit site therefore breaks the graph at its gate read, and again at the hand-over.
- **Profile consumers** — the `ProfileMixin` hooks an emitter calls, `_is_dynamic_energy_profile_active` and `_record_dynamic_energy`. They are ordinary traceable methods that reach the disabled gates, so the energy tensor an emitter builds stays in its caller's graph and only the query and the hand-over leave it. For the boundary this means a billing site carries a graph break wherever it is placed.

Nothing else is decorated. The unit forward, the engine `matmul`, the macro `vec_mat_mul`, the readout chain, and the digital aggregation are compile-friendly, not compiled.

## Why the boundary is drawn around the solve

Three properties of the solve decide it, none of them a property of the forward.

The solve is where compilation pays most. It settles the array through nested outer and inner Newton iterations, each assembling cell conductances, wire KCL residuals, and a coupled block-tridiagonal system — dense elementwise work that fuses well and dominates the runtime of an analog matmul. The projection is smaller but also runs once per chunk; compiling it lets topology-specific elementwise work and reductions fuse while its source keeps physically meaningful intermediate names.

The chunk loop cannot live inside a graph. Its trip count follows from the call's leading extent and the chunk size, both runtime values, so tracing it would either specialize the graph on one count or unroll an unbounded region. Chunking is not optional: the leading is the product of the caller's batch axes and the macro's own placement axes, and solving it in one block materializes every instance's Newton state at once.

The per-chunk solver and projection are the shape-stable points in the stack. Chunk slicing flattens whatever the caller's leading was into one `(effective_chunk_size, col, row)` cell grid. The effective size is the configured bound for a call at least that large and the complete leading extent for a smaller call; this prevents a small workload from paying for an oversized padded solve while keeping every chunk of a large workload on the configured shape. Above those leaves, the leading shape and rank vary per layer, placement axes are size-1 under some mappings and not others, and every billing site breaks the graph; self-compiling that forward yields many small frames, each re-keyed by geometry that `dynamic=True` cannot absorb, since a size-1-to-N flip or a rank change is not a dynamic dimension. The stable configured shape is what makes the usual large-call cost affordable: the block-tridiagonal sweep unrolls linearly in the row count, so its graph is deep and its cold compile long, but one main compile signature is shared by all full chunks. A smaller call may specialize one additional leaf shape to its actual leading extent instead of executing padded positions up to the configured bound.

## Recompile keys are wider than shape

`dynamic=False` pins the leaf to a static shape, but shape is not the whole cache key: the Python type of every argument enters it, and for a scalar argument so does the value. Exceeding the recompile budget for one code object does not raise — the leaf falls back to eager and the run is merely slow.

- The two rail resistances arrive as Python floats, so each distinct value compiles its own graph. Sweeping resistance or geometry under compilation is what exhausts the budget; calibration tools and the test suite therefore run under `TORCH_COMPILE_DISABLE=1`.
- Snaps arrive as frozen dataclasses, so their PyTree structure is part of the key. An optional field present on one call and absent on the next re-keys the leaf.
- The cell model and both clamp drivers arrive as module arguments. Whether every macro layer shares one graph or each cold-compiles depends on their state being lifted as guarded graph inputs rather than pinned per instance; the symptom of the latter is a burst of cold compiles at model build instead of one.
- A `solve_chunk_size` of zero hands the whole leading to the leaf unpadded, so the leaf then sees one shape per call site rather than one shape overall.

## Compiled peak memory exceeds eager

Chunking bounds the peak of the whole call, which makes `solve_chunk_size` the single knob over it — and the knob has to be tuned against the compiled path. Inductor co-allocates a graph's intermediates where eager Python frees them as it goes, so the compiled per-chunk peak exceeds the eager one at the same chunk size, and a value proven in an eager run can run out of memory once the leaf compiles.

## Tail chunks are padded

A short final chunk is padded up to the call's effective chunk size by repeating its last leading coordinate, which is how every chunk of that call reaches the leaf at one shape. The effective size is `min(solve_chunk_size, total_leading)`, so a whole call smaller than the configured bound needs no padding. A padded position is a duplicate of a real one, and duplicates are dropped only when the fold writes the projected result back to its global indices. A projection therefore keeps the chunk axis intact and reduces only over the axes behind it; a reduction across the chunk axis would count the duplicated positions, which is how a padded tail turns into inflated energy or another aggregate rather than into a shape error.

# Crossbar DC solve

One array solve crosses four owners: the array holding the geometry and the physical state, the chunk loop bounding the call's memory, the numerical solver leaf, and the two boundary clamps the solve evaluates as it iterates. Each owns a disjoint part of the call, and the seams between them are where changing one end obliges the maintainer to read the other.

## Ownership across one solve

- **The array** owns geometry, the lattice constants, the cell sub-module, and the measurement. It computes the call's broadcast leading, snapshots the cell once at the full call shape, makes exactly one chunked solve call at its original shapes, and reassembles what comes back. It holds no chunking logic and no numerical state.
- **The chunk layer** owns memory tiling and nothing else. It is signature-agnostic — a snap argument is sliced, every other argument passes through — so it encodes nothing about the solve it wraps and no circuit meaning of the axes it slices.
- **The solver** owns the method and no state. It is a plain tool object rather than a module, so the fabricate cascade, `state_dict`, and device migration never sweep it in, and every actor it needs arrives per call. The array selects it as an explicit code dependency — one array topology has one numerical implementation — so nothing on this path is registry dispatch; [construction](construction.md) covers what is.
- **The cell** owns the device stack. It condenses its own internal node and presents one signed two-terminal branch, so the array system carries no per-cell unknown, the solver opens no device leaf, and no device current is stamped outside the cell. The sign contract on the two terminal conductances is consumed unchecked: a cell returning the wrong sign corrupts the solve silently rather than raising.
- **The boundaries** reach the solve as a structural role: the transfer law alone, evaluated at a port current against a snap the caller has already sampled. Sampling and delivery both stay outside the role, so a driver, a switched capacitor, or an amplifier held in clamp satisfies the same surface while each is constructed, snapshotted, and billed by the block that owns it.

Both rails' rest levels come off those same snaps. The nominal reference the role requires is the ideal level and carries no driver-owned offset or draw: the solve seeds its warm start from it, and the array measures capacitive displacement from it. That is why a concrete driver keeps its perturbations in fields of its own — billing against a perturbed reference would charge a driver's non-ideality to the array's capacitance.

## Why the solve is nested

The formulation — an outer clamp Newton around an inner coupled wire Newton, with every cell condensed at each step — and its well-posedness are the [solver spec](../reference/primitive/xbar/solver/col_bl_col_sl.md). What the decomposition buys the software is the ownership split above: each sub-problem is small enough for one component to own. The cell's condensation keeps every device unknown inside the cell; the clamp stays a transfer law evaluated per step rather than a solved mesh node, which is what lets a boundary be a role instead of a member of the array's module tree; and the wire system stays block-tridiagonal in the row index, which is what keeps the per-chunk working set a plain product of chunk, column, and row extent. A simultaneous Newton over every unknown would fuse the three into one system no single component could own.

## The chunk loop is a memory knob

`solve_chunk_size` bounds memory and moves no number: a call's results are bit-identical across chunk sizes, with noise on as well as off. Every per-call draw is taken before the loop, once, at the full call shape — [physical_state](physical_state.md) owns that rule — so a chunk is a re-view of positions that already exist rather than a sampling event of its own, and slicing acts on the finished snap object. Treat a numerical difference between two chunk sizes as a bug.

What the knob bounds is the whole call's peak, not merely the solver's own transient. A chunk's grid-shaped state — the node voltages and the wire Jacobian that dominates it — dies with the chunk that produced it, and the fold reassembles only the small per-chunk measurement, so nothing grid-shaped outlives an iteration. Snaps cost nothing to hold at full shape: a broadcast field is a stride-0 view, and the slicing rule collapses, indexes, and re-expands, so a chunk materializes one value per real trailing position. That rule is mechanical and circuit-blind, which is what lets a caller state its correlation structure by expanding a drive onto the full grid and still pay only for what varies across it.

Two things make a tuned value non-portable. The compiled per-chunk peak exceeds the eager one at the same size ([compile](compile.md)), and a caller that flattens a serialized axis into the column axis multiplies the columns of a single solve, so a size proven under one scan organization is far too large under the other.

## The measurement belongs to the caller

The solve returns a grid-shaped operating point, and what survives a chunk is a modelling decision rather than a rank heuristic, so the caller hands the per-chunk measurement in and the layer folds its result alone. Reassembling the operating point instead would carry every node grid at the whole leading. A boundary quantity the wrapped solve does not declare travels beside it as a bare tensor, sliced by the same rule and reaching the measurement alone: a boundary with no fabricated state of its own, such as the word-line drive, needs no snap to be seen.

The port state is folded back to the full leading before anything is delivered on it, because the caller bills each boundary on that boundary's own column instance axis, whereas a per-chunk delivery would bill it against the ravelled chunk axis. A per-chunk measurement keeps its chunk axis intact and reduces only over the axes behind it — tail padding puts duplicate positions on that axis, which a reduction across it would count ([compile](compile.md)).

The solver entry point is also usable on its own — no array, no chunk loop, one fixed shape — for a caller that wants the raw operating point the measurement folds away. Both paths settle the same system and must agree numerically.

## Probing a solve

A solve's iteration trajectory leaves through a co-located recorder, never through the return value. The emitter stays blind to who is listening: the eager wrapper reads the recorder's active slot once and hands the answer to the compiled leaf as a plain flag, the leaf emits the whole trajectory whenever that flag is set, and the recorder turns away on arrival what its own threshold does not want. Both placements are load-bearing — reading the slot inside the leaf would bake one answer into the graph, and gating at the emit site would specialize the graph on a recorder's policy — and [compile](compile.md) holds the gate mechanics.

Nothing on the emit path feeds back into the iteration, so a solve returns the same operating point bit-identically whether or not anything listens; what a probe changes is cost. A probed solve breaks the graph at every emission and retains one residual grid per rail per inner step for as long as the recorder is open, both scaling with the iteration counts, which is why production paths run unprobed. The stream carries the residuals rather than the iterates they were evaluated at — the residual is what a convergence or calibration consumer reads — so a probed solve retains a residual pair per step instead of the whole working state of every iterate.

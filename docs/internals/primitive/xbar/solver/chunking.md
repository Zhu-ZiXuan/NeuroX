# Broadcast-leading chunking

The leading-axis machinery a chunked solve runs on: `iter_chunks` partitions the leading into fixed-size chunks (tail-padded, so the compiled solver body sees a single input shape), `slice_tensor` selects one chunk's positions out of a tensor with `slice_snap` as its dataclass-walking form, and `MeasureFold` folds the per-chunk measurements into one full-leading result.

## Design decisions

- **One traversal primitive under every dataclass-of-tensors rebuild.** `slice_snap`, the wrapper's operand collection, and `MeasureFold` all reduce to one shared walk, `walk_tensor_fields`: rebuild a frozen dataclass with a per-tensor transform, recursing into nested dataclass fields and carrying everything else through. Hand-rolling them would be four places for the same recursion, the same `None` skipping, and the same `dataclasses.replace` to drift apart; a read-only walk reaches the same primitive by giving the transform a side effect and discarding the rebuilt result.
- **Slicing is collapse, index, re-expand.** For each tensor field: narrow every trailing axis of stride 0 to size 1; drop the coordinate of every leading axis of stride 0, which holds one value for the whole call; apply one advanced index with the coordinates that remain; re-expand the axes collapsed in the first step. A field whose leading is stride 0 throughout keeps a size-1 chunk axis, so every sliced field presents the same leading rank. The rule is mechanical and circuit-blind: a word-line drive at `[batch, col*, row]` (stride 0 on `col`) costs `chunk * row` elements per chunk rather than `chunk * col * row`, a programmed cell buffer held across the call costs `col * row`, and a constant expanded over both costs one element — no caller has to order its expands to get that.
- **The fold preallocates and holds no per-chunk list.** `MeasureFold` allocates each output field once, flat at the whole leading, from the first chunk's trailing shape; `write` then puts each chunk into its global positions in place, and `result` restores the leading shape at the end. Nothing is concatenated and no chunk result outlives its iteration, so peak memory is the fold plus one chunk rather than twice the retained state. Allocating, writing and reading back are valid in that order alone, so they are one object's lifecycle rather than three functions a caller sequences by hand. Allocation and read-back are `walk_tensor_fields` rebuilds and `write` is two of its walks run in lockstep — one collecting the fold's own buffers in field order, the other scattering the chunk's tensors as it visits them in the same order.

## Contracts & invariants

- **Every chunk of a call agrees on field presence.** The lockstep walks behind `MeasureFold.write` pair the fold's buffers with the chunk's tensors by visit order, so the pairing invariant is field *presence* and not merely a shared type: an optional field absent in one chunk and present in another would misalign the two walks and scatter into the wrong buffer. Agreement on presence is what keeps an absent optional field, such as an energy nobody asked for, absent throughout.
- **The output side needs no leading.** A measurement carries exactly one leading axis, the chunk axis.

---

- **Reference**: N/A — a memory-tiling layer with no physics twin
- **Implementation**: `neurox/primitive/xbar/solver/chunking.py`
- **Tests**: `tests/primitive/xbar/test_chunking.py`

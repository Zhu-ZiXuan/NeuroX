# Tensor groups

`neurox/common/tensor_fields.py` holds `walk_tensor_fields`, the traversal every dataclass-of-tensors rebuild in the codebase runs on, and `neurox/common/tensor_group_mixin.py` holds `TensorGroupMixin`, which turns that traversal into a small shape-operation surface a snap class inherits. The extension contract lives in the class docstring beside the code; this page holds the reasoning behind it.

## Design decisions

- **One traversal, four call sites.** Rebuilding a frozen dataclass by mapping its tensor fields is not one subsystem's idea: chunk slicing, operand collection, measurement folding, and record detaching all do it. Written four times it is four copies of the same recursion, the same skip rule for non-tensor fields, and the same `dataclasses.replace`, each free to drift. `walk_tensor_fields` is that body once, and every one of those sites is a transform passed into it. A read-only walk needs no second primitive: give the transform a side effect and discard the rebuilt result.
- **A mixin, not a Protocol.** The surface here is inherited behaviour, not a capability a caller checks for. A Protocol expresses "anything with these methods will do", which is right for a role a consumer looks up — the clamp transfer, for instance — and wrong for shared code every host would otherwise copy. So the roles in this codebase stay structural Protocols and this stays an ordinary mixin, and the two never compete: a snap class can satisfy a Protocol and inherit this at the same time, which is exactly what a driver snap does.
- **The same-shape invariant is stated and never checked.** Every tensor field of a host is assumed to carry one shape, which is what makes a single `shape` or a single `dim` argument meaningful for the whole group. Enforcing it would mean a per-call scan of every field on a path that runs inside the solve loop, to catch a class-authoring mistake that a shape error in the very next operation surfaces anyway. The invariant is therefore a documented obligation on the host rather than a runtime gate, in line with the package-wide rule that only checks preventing silent corruption survive.
- **Dims follow `Tensor`'s own convention.** `flatten_axes` and `index_select` pass their dims through untouched, so negatives count from the right exactly as they do on a bare tensor. Normalizing or rejecting them here would make the group behave differently from the tensors it holds, for no gain.
- **The four methods are the ones a producer actually performs.** They are not a general tensor surface: `map_tensors` is the core and the other three are the shape operations a macro performs on a whole snap when it inserts, merges, or selects along an event axis — expand a sampled group onto the event shape, fold an inserted axis back into the one it merges with, or pick positions out of it. A fifth operation is added when a producer needs it, not in anticipation.
- **The chunking layer deliberately does not dispatch on it.** Chunk slicing classifies its arguments by a structural predicate — a frozen dataclass is a snap — and walks whatever it finds with `walk_tensor_fields` directly. Requiring the mixin instead would make chunkability an inheritance decision: a group authored elsewhere, or one that never needed the shape surface, would silently stop being sliced. The traversal is the shared thing; the mixin is a convenience on top of it, and the layer below takes the traversal.
- **Inheritance follows use, not kind.** A frozen dataclass of tensors carries the mixin where something actually reshapes the group; a group that only ever travels whole does not, because inheriting a surface nobody calls only widens the type's public API.

## Gotchas

- **`expand` is `Tensor.expand`, so it broadcasts and does not copy.** A group expanded onto an event shape is a set of stride-0 views costing one element each, which is what makes sampling at the full shape affordable — and equally what makes an in-place write to an expanded field corrupt every position at once.
- **A host that breaks the same-shape invariant fails later, elsewhere.** `flatten_axes(0, 1)` on a group whose fields disagree in rank succeeds on some fields and raises on another, or worse, succeeds on all and leaves the fields out of correspondence. Treat a shape surprise downstream of a group operation as a field-shape defect in the host.

---

- **Reference**: N/A — software utility
- **Implementation**: `neurox/common/tensor_fields.py`, `neurox/common/tensor_group_mixin.py`
- **Tests**: `tests/common/test_tensor_group_mixin.py`

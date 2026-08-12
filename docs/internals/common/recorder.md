# Recorder

`neurox/common/recorder.py` is the side channel every subsystem records through: `RecordBase`, the per-call frozen container an emitter fills, and `RecorderBase`, the context manager that collects one family's records. The extension contract lives in the class docstrings beside the code; this page holds the reasoning behind it.

## Design decisions

- **One mechanism, one contribution per subsystem.** PPA accounting and circuit diagnostics want the same thing from a side channel: a per-call container an emitter hands over, collected only while someone is asking, and never threaded back through call signatures. Written once per subsystem that would be several activation protocols, detach rules, and device policies free to drift; written once it is a class pair, and a subsystem contributes a record's fields and a family root and nothing else.
- **A record declares fields, and the base supplies the rest.** `__init_subclass__` turns each subclass into a frozen, keyword-only dataclass, and a hand-written `__init__` or `__post_init__` is refused at class-definition time — construction belongs to the base, so a record's call sites stay readable as its field list grows and nothing downstream rewrites a measurement. `detach` and `to` are one field walk ([tensor groups](tensor_group.md)) rather than a pair each family writes out; the walk recurses into nested dataclasses and rebuilds only when a field actually changes, so an already-detached, already-parked record costs no copy.
- **A record equals only itself.** `eq=False` on the base's own decorator and on the per-subclass application alike, so `==` and `hash()` are identity throughout the hierarchy. A record is one measurement event, not a value: two calls that happened to measure the same numbers are two events, and merging them in a set or a comparison would silently lose one. The base's own decorator is what makes this hold — a generated zero-field `__eq__` there would be inherited and would call any two records of a class equal.
- **The family, not the class, is the unit of activation.** A direct subclass of `RecorderBase` opens a family and owns the active slot every class below it shares. A family root and one of its own subclasses are therefore mutually exclusive collectors, while independent families collect side by side without knowing about each other. `RecorderBase` itself opens no family, so it can neither collect nor be asked: it raises rather than fabricating a fallback slot that would silently swallow every submission.
- **One open book per family.** Entering a second recorder of a family raises instead of nesting. A record belongs to one book; copying it into several would put one measurement in every one of them, and any consumer that later summed two books would double-count it. A narrower window is a separate measurement over its own stimulus, not an inner frame of a wider one.
- **The book belongs to the instance.** Re-entering one recorder accumulates into the same list, and a fresh book is a fresh instance. A run measured over several context blocks therefore totals as one book without the mechanism carrying a merge operation, and nothing an earlier recorder collected can leak into the next one.
- **Only a clean exit closes a book.** `__exit__` frees the slot unconditionally, so a raising body never leaves the family latched, but finalization runs only when the body did not raise: a failed measurement keeps its records exactly where they were recorded rather than paying a sweep whose own failure could mask the original error.
- **Where records rest is the harness's call.** The recording device is the emitter's, and a recorder takes the device its records are parked on as a construction argument — a harness knob rather than a physical quantity, which is why it carries a code default. The sweep visits the whole book on every clean exit and returns an already-parked record unchanged, so it needs no new-versus-old bookkeeping. Each record parks itself, so a cross-device book of $N$ records costs $N$ transfers: a consumer collecting a large book on an accelerator, or one that post-processes records where they were recorded, passes `device=None` and keeps them in place.
- **The slot is reached from outside every graph.** The demand gate, the slot read, and the hand-over are all `@torch.compiler.disable`. The slot is Python state a trace cannot guard on: Dynamo folds an empty slot into the graph as a constant and installs no guard, so a region first compiled outside any context would stay pinned to "inactive" and silently collect nothing ever after. The graph breaks are the design; everything an emitter computes to build a record stays inside its own graph.
- **An unclaimed record is dropped, not an error.** Submitting with no active recorder returns. An emitter's gate is read at a graph break and a helper may submit unconditionally, so an unclaimed record means nobody was asking — not that the measured computation failed.

## Contracts & invariants

- **Detach at the hand-over.** `submit` stores `record.detach()`, so no autograd history leaves the caller's graph through the side channel and a book never keeps a graph alive.
- **Submission count is the link's own contract.** The mechanism records every `submit` call exactly once in the family's active recorder; what a submission means and how often it happens belong to the module that defines the record.
- **Collection is single-threaded.** The active slot is a plain class attribute, not thread-local; two threads recording at once share one book.
- **A record captured under CUDA-graph capture is cloned first.** Under `torch.compile(mode="reduce-overhead")` a submitted tensor may live in cudagraph-owned memory that a later replay overwrites, so an emitter inside such a region clones before it submits.

## Performance & resources

Per submission: one `detach` field walk, which rebuilds nothing when no field carries a graph, plus one list append — no device sync. Finalization costs one visit per record in the book and one device transfer per record when a device is declared, which `device=None` skips entirely. Memory grows linearly with the book, at the full size of what each record carries.

## Gotchas

- **A book read inside its own context has not been parked.** The list is live, so records read there sit on the device they were recorded on, whatever the recorder declares. Read after the block.
- **Two recorders of one family do not nest.** Scope the contexts side by side; an inner window that needs its own book is its own measurement.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/recorder.py`
- **Tests**: `tests/common/test_recorder.py`

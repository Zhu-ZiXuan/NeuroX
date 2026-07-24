# Prober

`neurox/common/prober.py` is pure mechanism: a `SupportsDetach` payload protocol plus an abstract `Prober(Generic[PayloadT])` base requiring each subclass to provide its own typed LIFO active stack, context-manager scoping, a `.records` list, and two classmethods — `active()` (demand predicate) and `submit(payload)` (single central detach, shared across every active prober of that subclass). The mechanism carries no channels, no allowlists, and no per-module wiring.

## Design decisions

- **One concrete `Prober` subclass per observation link.** A link consists of one emission contract and one payload type. Each subclass is the sole capture channel for its link; there is no shared catch-all prober.
- **Link definitions are colocated.** A concrete prober, its payload, and its emitter belong in the same module. The common mechanism therefore knows neither the available links nor their payload fields.
- **Each link subclass explicitly owns one typed active stack.** The generic base cannot type a class variable in terms of `Self` or `PayloadT`, so each concrete link declares `_active_stack: ClassVar[list[Prober[ConcretePayload]]]` and returns it through the abstract `_stack()` classmethod. A stack receives only submissions made through its own subclass.
- **Emitters require no inheritance or registration.** An emission checks the concrete prober's `active()` predicate before calculating diagnostics or constructing a payload, then calls `submit(...)`. The guard keeps inactive observation work out of the numerical path.
- **Probers stack; a session nests inside a narrower capture.** Activation is a class-level LIFO stack per subclass, so an outer session-scoped prober keeps recording while an inner one captures a narrow window, and an emission reaches every active prober of the emitted link's subclass.
- **Records are stored full, detached, on the recording device.** A probe's value is the payload's tensors themselves, so `submit` detaches once — never reducing, syncing, or copying to host — and shares the same frozen object across every active prober of the subclass; a probing run holds every submitted payload alive until the prober is dropped.
- **No emitting module or name is stored.** A record is the payload alone.
- **The abstract base is not instantiable.** `_stack()` is the required abstract classmethod, so a link that does not provide typed storage cannot be instantiated. Calling `active()` or `submit()` on the abstract base reaches `_stack()`'s `NotImplementedError` rather than a fabricated fallback stack.

## Contracts & invariants

- **Submission count is link-defined.** The common mechanism records every `submit(...)` call exactly once in each active prober of that subclass. The module defining a link owns the meaning and frequency of those submissions.
- **LIFO discipline.** `__exit__` pops its own frame and raises if the stack top is not `self`; `with`-block usage guarantees this. The stack is class-level and process-wide, not thread-scoped.

### Public API

- `SupportsDetach` — the payload `Protocol`: `detach(self) -> Self`.
- `Prober[PayloadT]` — generic abstract base; a link subclass binds `PayloadT` to its own observation type.
- `SomeLinkProber._stack()` — returns that link's explicitly declared typed active stack.
- `with SomeLinkProber() as prober:` — enters the link's active stack; `prober.records` accumulates for the block's duration.
- `SomeLinkProber.active()` — `@torch.compiler.disable`; `True` iff some prober of that subclass is currently active. The guard an emitter checks before building and submitting a payload.
- `SomeLinkProber.submit(payload)` — `@torch.compiler.disable`; no-op on an empty stack, else detaches `payload` once and appends the same object to every active prober's `.records`.

## Performance & resources

With no active prober, `active()` returns `False` and the guarded diagnostic computation and payload construction never run. `@torch.compiler.disable` keeps the guard outside an enclosing compiled graph. Per emission with $k$ active probers of the link's subclass, the mechanism performs one `detach()` and $k$ Python appends; no device sync occurs. Memory grows linearly with admitted records.

## Gotchas

- **Do not probe an unbounded run.** Records are full payloads; a long capture accumulates them all. Scope the `with` block to the stimulus batch being fitted.
- **A prober sees only its subclass's link.** Different concrete prober subclasses have independent stacks even when their contexts overlap.

## Known limitations

- **No persistence.** Records live in process memory only; a calibration tool serializes what it needs.

---

- **Reference**: N/A — software observation mechanism
- **Implementation**: `neurox/common/prober.py`
- **Tests**: `tests/common/test_prober.py`

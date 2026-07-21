# Prober

## Summary

`neurox/common/prober.py` is pure mechanism: a `SupportsDetach` payload protocol plus an abstract `Prober(Generic[PayloadT])` base requiring each subclass to provide its own typed LIFO active stack, context-manager scoping, a `.records` list, and two classmethods — `active()` (demand predicate) and `submit(payload)` (single central detach, shared across every active prober of that subclass). The mechanism carries no channels, no allowlists, and no per-module wiring.

## Design decisions

- **One concrete `Prober` subclass per observation link.** A link is one emitter site plus its payload type — `SolverProber`, `XbarCell1t1rDetailProber`, `CurrentAdcProber`, `VoltageAdcProber`. Each subclass is the sole capture point for its link; there is no shared "any channel" prober.
- **Co-location is the law: a link's `Prober` subclass lives in the same file as its payload class, next to the emitter.** `SolverProber` + `SolverObservation` sit in `solver/nested.py` beside `NestedParallelRailSolver`; `XbarCell1t1rDetailProber` + `XbarCell1t1rDetailObservation` sit in `cell/_1t1r_detail.py`; `CurrentAdcProber` / `VoltageAdcProber` sit in their family's `base.py` beside the family's `Observation`. The observation contract — what a link captures and at what level of generality — is authored once, at the emitter, not split across a mechanism module and a call site.
- **Each link subclass explicitly owns one typed active stack.** The generic base cannot type a class variable in terms of `Self` or `PayloadT`, so each concrete link declares `_active_stack: ClassVar[list[Prober[ConcretePayload]]]` and returns it through the abstract `_stack()` classmethod. Entering a prober of one subclass never captures an emission another subclass submits; that isolation is the whole routing mechanism.
- **Emitters inherit nothing and declare nothing.** There is no allowlist to satisfy and no per-module wiring to set up. The idiom at every emission site is `if XxxProber.active(): <build payload>; XxxProber.submit(Payload(...))` — demand-gated so an unsubscribed run pays nothing beyond the boolean check, and both the diagnostic computation and the payload construction sit inside the guard.
- **Probers stack; a session nests inside a narrower capture.** Activation is a class-level LIFO stack per subclass, so an outer session-scoped prober keeps recording while an inner one captures a narrow window, and an emission reaches every active prober of the emitted link's subclass.
- **Records are stored full, detached, on the recording device.** A probe's value is the payload's tensors themselves, so `submit` detaches once — never reducing, syncing, or copying to host — and shares the same frozen object across every active prober of the subclass; a probing run holds every submitted payload alive until the prober is dropped.
- **No emitting module or name is stored.** A record is the payload alone.
- **The abstract base is not instantiable.** `_stack()` is the required abstract classmethod, so a link that does not provide typed storage cannot be instantiated. Calling `active()` or `submit()` on the abstract base reaches `_stack()`'s `NotImplementedError` rather than a fabricated fallback stack.

## Contracts & invariants

- **Emit at most once per logical operation per link, unless the caller documents otherwise.** A driving solver's nested inner steps are the one documented exception — see the solver and Detail-cell docs for their own per-call emission counts.
- **LIFO discipline.** `__exit__` pops its own frame and raises if the stack top is not `self`; `with`-block usage guarantees this. The stack is class-level and process-wide, not thread-scoped.

### Public API

- `SupportsDetach` — the payload `Protocol`: `detach(self) -> Self`.
- `Prober[PayloadT]` — generic abstract base; a link subclass binds `PayloadT` to its own observation type.
- `SomeLinkProber._stack()` — returns that link's explicitly declared typed active stack.
- `with SomeLinkProber() as prober:` — enters the link's active stack; `prober.records` accumulates for the block's duration.
- `SomeLinkProber.active()` — `@torch.compiler.disable`; `True` iff some prober of that subclass is currently active. The guard an emitter checks before building and submitting a payload.
- `SomeLinkProber.submit(payload)` — `@torch.compiler.disable`; no-op on an empty stack, else detaches `payload` once and appends the same object to every active prober's `.records`.

## Performance & resources

With no active prober, `active()` returns `False` and the guarded diagnostic computation and payload construction never run — no tensor is touched, and `@torch.compiler.disable` keeps the guard (and the branch it collapses at trace time) out of any caller's compiled graph. Per emission with $k$ active probers of the link's subclass: one `active()` check, then — only when it is `True` — the payload build plus one `detach()` and $k$ Python appends; no device sync ever happens on the emit path. Memory grows linearly with admitted records.

## Gotchas

- **Do not probe an unbounded run.** Records are full payloads; a long capture accumulates them all. Scope the `with` block to the stimulus batch being fitted.
- **A prober only ever sees its own subclass's link.** Opening `SolverProber()` reaches nothing an `XbarCell1t1rDetailProber` emits, even inside the same `with` block; open every link a consumer needs as its own `with`-bound name.

## Known limitations

- **No persistence.** Records live in process memory only; a calibration tool serializes what it needs.

---

- **Reference**: N/A — the prober has no physics spec; the calibration procedures consuming it live with their tools
- **Implementation**: `neurox/common/prober.py`
- **Tests**: `tests/common/test_prober.py`

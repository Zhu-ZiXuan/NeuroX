# Prober

A prober is a [recorder](recorder.md) family that captures one emission link whole, for calibration and diagnostics rather than for accounting. The common layer holds no prober class and knows no link: a link is one emission contract plus one record type, and both are declared beside the code that emits them.

## Design decisions

- **One prober family per link, colocated with it.** A link's record, its prober, and its emitter live in the same module, so adding one is two class declarations beside the emitting code and nothing else — no registry, no allowlist, no per-module wiring, and no shared catch-all prober that would have to know every field any link might carry. Each family is the sole capture channel for its own link, and a submission never reaches another family's book.
- **An emitter needs no inheritance and no registration.** The emission site reads its link's demand gate before it computes any diagnostic or builds any record, then submits. Nothing about the emitter changes when nobody is collecting: the returned numerical result is identical probed or unprobed, because everything the gate guards is diagnostic-only.
- **A record is captured whole, on the device it was computed on.** A probe's value is the tensors themselves — residuals, inputs, decided codes — so nothing is reduced, summed, or synced to the host on the way in. A consumer fitting a model over a large book keeps it where it was computed and reduces on its own terms.
- **A record identifies the event, not the emitter.** Which instance produced it is the collecting caller's own knowledge: a calibration drive knows what it drove, and the alternative — carrying a module reference — would keep the whole tree alive behind every captured tensor. Where the emitter is not a module at all, the record carries the fixed name of the entry point that emitted it.
- **A calibrated constant is not a measurement.** A record carries what the call measured or decided; a value the emitter was configured with stays out of it, because it is already readable from the config that supplied it.

## Contracts & invariants

- **The demand gate is what keeps diagnostics out of the numerical path.** An emission site computes its diagnostic only inside the gate, so an unprobed run pays nothing but the gate read itself.
- **Submission count is the link's own contract.** The module defining a link owns what one submission means and how often it happens; the mechanism records each of them once. A consumer that pairs two streams positionally depends on that frequency, so a link states it where it is declared.

## Performance & resources

With no prober active, the gated diagnostic computation and the record construction never run, and the gate read is a graph break rather than device work. Per capture the mechanism performs one `detach` and one append, with no device sync. Memory grows linearly with the book, at the full size of the captured tensors.

## Gotchas

- **Do not probe an unbounded run.** Records are whole tensors and nothing prunes the book; scope the context to the stimulus being fitted.
- **A prober sees only its own link.** Overlapping contexts of different families capture independently, so a consumer pairing two streams pairs them itself.

## Known limitations

- **No persistence.** A book lives in process memory only; a calibration tool serializes whatever it needs from it.

---

- **Reference**: N/A — software diagnostic mechanism
- **Implementation**: `neurox/common/recorder.py`, plus each link's own module
- **Tests**: `tests/common/test_recorder.py`

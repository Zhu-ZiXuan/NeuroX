# Non-ideality kernels

Every perturbation a physical model applies — a fabrication mismatch, a programming deviation, a per-access fluctuation — is drawn by one shared kernel library rather than by the model that owns the source. The split is narrow and load-bearing: a kernel owns the distribution and nothing else, and the calling module owns every fact about the event that produced the draw. Maintaining either end means knowing which half of that contract it holds.

## One vocabulary for every perturbation

The library is the single place randomness enters a physical model. A subsystem parameterizes an existing kernel instead of writing a draw of its own, so each statistical law the [non-ideality spec](../reference/primitive/nonideality.md) states has exactly one implementation, and two subsystems quoting the same law compute it identically rather than nearly so.

The set stays closed because the kernels are keyed by where the spread comes from, not by which circuit uses them: a spread fixed by configuration and a spread derived from the perturbed value each have a named kernel, while a source whose spread follows per-call operating state computes that value itself and hands it to the plain additive kernel. Sigma arithmetic that depends on an operating point therefore lands in the module that knows the operating point, and never as one more near-duplicate kernel.

## The kernel draws; the caller decides when and over what

A kernel sees a tensor, a spread, and a flag. It cannot distinguish a static draw from a dynamic one — the same additive Gaussian serves a fabrication mismatch held for the life of an instance and a fluctuation redrawn at every access. What makes a source static or dynamic is the lifecycle hook that calls it, and what makes it per-instance or per-access is the shape of the tensor handed in; [physical_state](physical_state.md) owns both rules, and the calling module honors them.

Two consequences follow for anyone adding a source.

- **The kernel adds no axis and removes none.** It draws elementwise at the shape it is given, so a caller wanting one deviation shared across positions hands in a tensor that does not span them, and a caller wanting independent deviations expands first. Correlation structure is a statement the caller makes with shape, never a service the kernel provides.
- **Nothing validates the classification.** No check confirms that a state-dependent source received a state-derived spread, or that a source held static was called from the fabrication hook. A misplaced call yields a plausible number rather than an error, so the defect surfaces only as a wrong distribution — which is why the placement is settled at the call site against the classification the subsystem's own specification states.

## The enable flag is a trace-time constant

Which sources are active is a per-run decision, carried as plain `bool` fields loaded before construction ([construction](construction.md)) and passed to each kernel as a keyword-only flag. Because those flags are Python values settled before the first call, a kernel's branch resolves while the graph is traced: a run with a source off holds no mask, no draw, and no allocation for it, and a run with it on introduces no data-dependent control flow for the compiler to break on ([compile](compile.md)).

The identity return is what makes an off source genuinely free. A disabled kernel hands back the object it was given, so a chain of disabled sources over a broadcast view materializes nothing at all. The cost is a rule every caller obeys: a kernel result is read-only. An in-place write on one reaches the module's own state when the source is off and a private copy when it is on, so the two policies diverge in what they corrupt rather than only in what they perturb.

## Configs sit beside their kernels

A kernel whose spread needs several coupled parameters declares its configuration in the same file as the function that consumes it, because the parameter set is part of the distribution's definition. A subsystem modelling such a source embeds that shared configuration as a field of its own rather than restating the fields, so one definition of what the source is serves every subsystem and its validation runs wherever it appears. The alternative — parameters declared per subsystem — makes the same physical source drift apart between the subsystems that claim to model it.

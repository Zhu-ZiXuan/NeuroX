# Differential voltage ADC base

The abstract `DiffVadc` carries config-keyed construction, the `convert` template method, and the shared differential-voltage quantization surface.

## Design decisions

- **Family dispatch keyed on config and policy types.** `DiffVadc.from_config(...)` resolves `(type(config), type(policy))`; adding a concrete voltage ADC registers its concrete pair and never touches `from_config`. A mismatched pair fails before leaf construction.
- **Empty marker `DiffVadcPolicy`.** The base policy carries no switch; each concrete ADC declares its own `*Policy(DiffVadcPolicy)` with that topology's toggles, so the policy shape follows the chosen topology, not a union of all topologies.
- **The zero point is exposed, never folded in.** The ADC returns the raw unsigned code and exposes its zero point through `zero_offset(bits)`. The base fixes the raw `unsigned_range` contract and the zero-point accessor, not a signed output.
- **Duration is declared here; its per-op constant is leaf-defined.** `latency__ns(*, bits)` is abstract on `DiffVadc` — a conversion is the only thing this family spends time on, and the executed resolution is its whole input — while there is no `latency_per_op__ns` field on `DiffVadc` / `DiffVadcConfig`. Fixed-latency impls carry that field on their own config; parametric impls derive the window from `bits`. A base field would force a single latency shape on topologies whose latency model genuinely differs.
- **`convert` is a probe-emitting template method.** The base's concrete `convert` calls the abstract `self._convert_impl(...)` and then, only `if DiffVadcProber.active()`, builds and submits a `DiffVadcObservation` (the two input voltages, the returned `code`, and the plain `bits` scalar) to `DiffVadcProber` — demand-gated so an unsubscribed run never touches the returned code or builds the payload. The observation records the conversion event alone; a reference is a calibrated constant rather than a measured quantity, so it stays out of the payload — the same rule the current-ADC family follows.
- **Reference count is a circuit property, not an interface law.** How many taps a call carries is decided by the concrete converter's own circuit — one full-scale reference for a CDAC that divides it internally, a full threshold bank for a flash comparator bank — so the base declares and validates no tap count. A leaf that needs a particular width states it itself.
- **`DiffVadcProber` is co-located with its payload in this module.** `DiffVadcProber(Prober[DiffVadcObservation])` sits beside `DiffVadcObservation` in `diff_voltage_adc/base.py`, next to `DiffVadc`, the observation link's sole emitter.
- **Rounding follows `self.training`, no override flag.** Stochastic-versus-deterministic rounding is the standard `nn.Module` train/eval state; there is no constructor-time override knob, so behaviour is toggled the same way as the rest of the model.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete voltage ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **Per-call operating point.** `convert(v_pos__V, v_neg__V, *, v_refs__V, bits)` receives the injected reference taps — shape `[..., n_ref]` with the taps last, symmetric with the current family's `i_refs__uA` — plus the resolution. No ADC self-holds a reference, and the interface carries no operating-mode identity: the owner names a mode to its reference source, which returns the matching taps, so mode is consumed before the converter is reached.
- **Raw unsigned code output, clamped to the legal range.** Every `convert` clamps the raw bucket to `[0, code_num-1]` and returns it unshifted; the clamp guards against stochastic-rounding jitter pushing the bucket out of range. When `code_num = 2**bits` the raw range is `[0, 2**bits-1]`.

---

- **Reference**: [adc base](../../../../reference/primitive/analog/diff_voltage_adc/family.md)
- **Implementation**: `neurox/primitive/analog/diff_voltage_adc/base.py`
- **Tests**: `tests/primitive/analog/test_adc_family.py`, `tests/primitive/analog/test_diff_voltage_adc_probe.py`

# Transcoder — Implementation

## Summary

The transcoder layer is the `Transcoder` ABC (`base.py`), which holds the shared positional state and the encoding-agnostic `decode`, plus one self-registering subclass per encoding (`true_form.py`, `complement.py`, `canonical.py`). Spec: [reference/mapper/transcoder/encodings](../../../reference/mapper/transcoder/encodings.md).

## Design decisions

- **Encoding selected by a string discriminator through the registry, not an `isinstance` ladder.** `Transcoder` parametrises `RegistryMixin[Encoding, "Transcoder"]`; each subclass self-registers with `@Transcoder.register_key("...")` and `Transcoder.create(encoding, radix=..., digit_num=...)` looks up the impl. Adding an encoding adds one file and one decorator and never touches the factory. The discriminator is a `Literal` (`Encoding`) so configuration and TOML carry the choice as a plain string that type-checks.
- **`decode` lives on the ABC; only `encode` and `value_range` are abstract.** The positional weighted sum is identical for every encoding (they differ only in the forward alphabet), so the shared reduction is written once on the base. Pushing it down to subclasses would duplicate it three ways and let them drift.
- **Construction inputs are exposed as read-only properties, not re-stored by callers.** `radix` and `digit_num` are `@property` because they are init-determined constants; `value_range` is a property for the same reason (derived from those constants). They are init-determined, so a property, not a method.

## Contracts & invariants

- **ABC observable surface.** A `Transcoder` exposes exactly: `encode(x, *, dim=-1) -> Tensor` (inserts a size-`digit_num` axis at `dim`), `decode(digits, *, dim=-1) -> Tensor` (removes that axis), and the `radix` / `digit_num` / `value_range` properties. `encode` and `decode` are mutual inverses *only within* `value_range`; outside it the forward map wraps and the round-trip is not recoverable - the value range is a caller contract, not an enforced clamp.
- **`encode` is shape-agnostic in `dim`.** The digit axis is inserted at the caller-chosen `dim`; the implementation must not assume a trailing axis. `decode` reduces whichever `dim` holds the digits and builds its positional-weight vector on the digit tensor's own device and dtype, so it stays correct across devices and never forces a host sync.
- **Validation is at construction.** `radix >= 2` and `digit_num >= 1` are checked in the base `__init__`; subclasses add no further construction validation.

## Performance & resources

- The encodings are `digit_num` division/remainder passes accumulated into a Python list, then one `torch.stack`. List accumulation (not in-place writes) keeps the unrolled loop fusable under `@torch.compile` at the caller. The work is integer-elementwise and negligible against the analog solve; there is no chunking or memory pressure at this layer.

## Gotchas

- **Value range is a contract, not a guard.** Encoding an out-of-range integer silently wraps or truncates - there is no error and no clamp. A caller must keep its inputs inside `value_range`; treating `encode`/`decode` as lossless for arbitrary integers is the anti-pattern.
- **Registration is an import side-effect.** A subclass is in the registry only once its module is imported; importing the `neurox.mapper.transcoder` subpackage loads all three. Reaching for a subclass through a deeper or partial import can leave the registry unpopulated. Import the ABC and `create` from the subpackage, not from `neurox.mapper` (which re-exports nothing).
- **Canonical carries across positions.** The canonical forward map mutates the running quotient with a carry while emitting each digit, so its per-digit step is not independent the way true-form's and complement's are; do not assume the three encodings share a digit loop body.

## Known limitations

- N/A.

---

- **Reference**: [signed-digit encodings](../../../reference/mapper/transcoder/encodings.md)
- **Implementation**: `neurox/mapper/transcoder/base.py`, `neurox/mapper/transcoder/true_form.py`, `neurox/mapper/transcoder/complement.py`, `neurox/mapper/transcoder/canonical.py`
- **Tests**: `tests/test_transcoder.py`
- **Decisions**: N/A — no ADR governs this module.

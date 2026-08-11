# Voltage reference

`Vref` is a leaf multi-output source holding a 2-D `[mode][tap]` bank and exposing it through the read-only `v_out__V`. It is structurally identical to `Iref`; only the dimension of the sourced quantity differs.

## Design decisions

- **PPA is static-only; nothing to log per call.** The reference emits no dynamic energy and reports no duration — the bias network stands and every tap is held at once — so its area and always-on bias power ride the `area_per_inst__um2` / `leakage_per_inst__uW` on its own config. There is no `v_supply__V` field and no bias-current field.
- **Fabricate-only: the source is an identity, not an event.** A reference generator has no notion of an access, so it has no `snapshot` and takes no per-call draw. What it owns is one static bank per fabricated instance, sampled once in `_sample_fabricate_mismatch` and held. Dynamic per-access variation belongs to the circuit that reads the tap, which is the block that knows what an access is and at what shape one call covers several — see [physical_state](../../physical_state.md).
- **`v_out__V` is a read-only property over the fabricated buffer, not a resolved selection.** It hands back the whole `[*inst_shape, mode, tap]` bank. Selecting a mode is the consumer's, because a mode means whatever the reading circuit says it means, and the existing mode-ordering law already lives there; resolving it here would push that meaning into the source and hide the bank layout behind a call that has to be made once per access instead of once per view.
- **Fan-out is by view.** One instance's identity reaching several positions of a call is a broadcast of the same numbers, so a consumer expands the property's output onto its own shape and pays one element for it. That is what makes the shared-identity semantics free, and it is also what would be destroyed by a per-call resample here.
- **The departure is relative (multiplicative).** `apply_relative_gaussian` scales by `1 + randn * sigma`, so one sigma applies uniformly to taps of differing magnitude and a `0` V tap stays exactly zero; an absolute per-tap sigma would need a tuple.
- **Static mismatch spans `inst_shape` and nothing more.** The fabricated taps are one physical generator per instance. A *static* per-reader offset belongs to the reader, which already fabricates its own offset over its own `inst_shape`, so the source deliberately carries no per-consumer static axis; adding one would double-count the same deviation.
- **One instance serves one consumer.** A reference source is fabricated for the circuit it feeds, so it never acts as a shared multi-consumer bank; a second consumer gets its own instance with its own static seat.

## Contracts & invariants

- **State lifecycle.** `_nominal_v_refs__V` is a non-persistent nominal buffer with shape `(mode_num, tap_num)`. `fabricate()` clone-expands it to `(*inst_shape, mode_num, tap_num)` and assigns `_v_refs__V` ordinary state, with tolerance applied when enabled. The clone keeps an in-place write on the fabricated state — or on a view of it — from reaching the registered buffer. Reading `v_out__V` requires fabrication first.
- **`mode_num` / `tap_num` are properties** on config and module (`len(config.v_refs__V)` / `len(config.v_refs__V[0])`) and define the bank geometry.
- **Range-checking `mode` belongs to the consumer** that owns the mode set, since the source publishes the whole bank and asserts nothing about which slice means what.
- **Modes are equal-length and non-negative**, validated at config time (`>= 1` mode, `>= 1` tap per mode, equal tap lengths, each tap `>= 0`). Ordering within a mode is deliberately **not** validated here: what a mode means is the consumer's knowledge, so a decision ladder's ascent is checked at config time by the converter or macro that wires it. The field is always an explicit 2-D bank, including for the degenerate single-mode single-tap `[[v]]`; a flat TOML array is rejected by deserialization.

## Gotchas

- **Tolerance is fixed by `fabricate()`, not by construction.** A freshly constructed reference has no `_v_refs__V` state; call `fabricate()` before reading `v_out__V`.
- **With tolerance off the property returns a view, not a copy.** `apply_relative_gaussian` returns its input untouched when disabled, so the fabricated state can stay an expanded view of a private clone of the nominal buffer. That is what keeps a wide fan-out free of a materialized allocation; never mutate the returned tensor in place.

---

- **Reference**: [voltage_reference](../../../reference/primitive/analog/voltage_reference.md)
- **Implementation**: `neurox/primitive/analog/voltage_reference.py`
- **Tests**: `tests/primitive/analog/test_voltage_reference.py`

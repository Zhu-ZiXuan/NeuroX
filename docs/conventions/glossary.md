# Glossary

The canonical term, symbol, and code name for each concept used across NeuroX subsystems — one concept, one expression.

## Operations and lifecycle

- **program** — write a weight into the tile as a non-negative integer digit tensor whose digits combine positionally with radix $r$; writes the actual weight state and may be re-called, each call overwriting the previous encoding.
- **fabricate** — resample static manufacturing mismatch across every owned module from the unchanged nominal template; re-callable with no state accumulating across calls, orthogonal to `program`.
- **snapshot** — the verb (the `snapshot()` method) that materializes a per-call `snap` from current dynamic noise; never persisted.
- **solve_dc** — solve a circuit's DC steady state under its boundary constraints, returning a per-call `dcop`; each solving module implements its own `solve_dc`.
- **snap** — the per-call frozen container a `snapshot` produces, holding one call's materialized dynamic noise; `*Snap` suffix. Never "transient"; write "per-call".
- **dcop** — DC operating point: the per-call frozen DC steady-state solution (node voltages / branch currents) returned by a `solve_dc`; `*Dcop` suffix.
- **observation** — the per-call frozen payload an emitter submits to its observation link, holding that call's diagnostic tensors; `*Observation` suffix.

## Value domain and slicing

- **value domain** — the integer grid a tile can physically carry: the input grid $\mathcal{X}$, the digit count $D$, and the radix $r$, all published by the xbar interface (the authority) for algorithm-side ranges to map onto. The algorithm-side `value_range` (the complete value an input scalar can take) is distinct from the primitive single-cell `digit_range`.
- **digit** — the level-0 value-domain unit: the integer symbol one xbar cell carries, in radix $r$ (`digit_radix`), bounded by the per-cell `digit_range`, and realized physically as the cell conductance state (see [device/rram](../reference/primitive/device/rram.md)). `digit_count` ($D$) digits make one slice.
- **slice** — the level-1 value-domain unit (synonyms weight-slice, one-digit slice): a fixed-capacity positional piece of a value made of `digit_count` digits, with positional ratio the slice radix $R$ (`slice_radix`). The slice count ($S_w$ weight side, $S_a$ input side) is config-given, not inferred. Slices recombine into the value by the radix-weighted shift-add. See [unit family](../reference/architecture/unit/family.md) and [digital/shift_adder](../reference/primitive/digital/shift_adder.md).
- **value** — the level-2, role-neutral algorithm scalar: a weight on the weight side, an activation on the input side, decomposed into slices for the macro and recombined from them. The xbar carries values role-neutrally and asserts no weight/activation application role.
- **encoding (codec)** — the generic integer-to-digit-string codec over (radix, `digit_count`, policy), selected by the discriminator `"true_form" | "complement" | "canonical"`; an implementation primitive with no slicer or application semantics. See [encoding](../internals/common/encoding/encodings.md).
- **coding scheme** — the scheme-xbar-level rule mapping logical columns to physical columns (data/digit ordering, any reference column) plus the readout combination that recovers the signed result; the concrete rule belongs to each scheme xbar.

# CimEngine family

The execution dimension of the CIM unit: an engine is the workload-agnostic slicing / tiling / macro-cycle / aggregation pipeline that places one integer matmul on [CIM macros](../../../../primitive/macro/cim/family.md) and recombines the per-macro reads. The engine variants differ only in how the precision-slicing axis maps onto the tiling layout; the operator a unit exposes ([family operator law](../../family.md)) is orthogonal.

- [direct](direct.md) — the no-slice corner $S_w = S_a = 1$: values map straight onto one tile's native value range.
- [inter_array_slice](inter_array_slice.md) — weight slices distributed across separate tile planes, recombined by cross-plane shift-add.
- [intra_array_slice](intra_array_slice.md) — weight slices gathered into adjacent columns of one tile, recombined by intra-tile shift-add.

The variant is selected by the concrete engine config nested inside the unit config (the `_neurox_class` discriminator); see the [config reference](../../../../../api/README.md).

## Governing laws

**Pipeline.** Every engine realizes the exact integer dot product

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k}$$

through the same stages: decompose values into positional slices (where present), tile the matmul across the physical grid ($T_r$, $T_c$), drive the per-tile macro reads, and aggregate — shift-add over the slice axes, plain accumulation over the contraction tiles, concatenate and trim to $N$ over the output tiles. Decompose and aggregate are inverse operations, per the [family contract](../../family.md).

**Input-phase generation.** The engine owns serialization required by
`max_active_num`. For a macro with $N_{\mathrm{in}}$ logical inputs and at
most $A$ selected positions per conversion, it expands each input into
$P=\lceil N_{\mathrm{in}}/A\rceil$ zero-masked phases. The phase axis is always
present and is inserted immediately left of the macro instance-aligned block.
The macro converts every phase independently and returns trailing
$[N_{\mathrm{out}}]$ codes.

**Two-stage digital accumulation.** The engine first reduces its input-phase
axis at each macro output port, then reduces the $T_c$ contraction-tile axis.
The two axes use distinct digital modules and are accounted separately.

**Axis layout.** Every variant's organized weight tensor places its present slice/tile axes in the canonical $[S_a, S_w, T_c, T_r]$ order ahead of the tile-owned trailing block; a variant that does not use an axis omits it entirely. The fixed order is what lets the aggregate reductions name their axes by a stable negative index.

## Noise & non-idealities

An engine adds no non-ideality of its own: the slicing and aggregation arithmetic is exact integer arithmetic. Every deviation enters through the tile reads — analog non-idealities and ADC quantization — specified by the [CIM macro family](../../../../primitive/macro/cim/family.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $S_w, S_a$ | weight-, activation-slice counts | — | `w_slice_num`, `x_slice_num` |
| $T_r, T_c$ | output-, contraction-axis tile counts | — | structure count |
| $N_{\mathrm{in}}, N_{\mathrm{out}}$ | macro logical input / output capacity | — | `input_num`, `output_num` |
| $A$ | maximum selected inputs per conversion | — | `cim_macro.max_active_num` |
| $P$ | input phases per macro read, $P = \lceil N_{\mathrm{in}}/A\rceil$ | — | `_input_phase_num` |

The logical dims, value-domain symbols, and the ADC surface are in [unit/family](../../family.md#symbols).

## References

TODO.

---

- **Internals**: [engine base](../../../../../internals/architecture/unit/cim/engine/base.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md)

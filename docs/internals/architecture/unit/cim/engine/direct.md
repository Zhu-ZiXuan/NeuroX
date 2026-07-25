# DirectCimEngine

The no-slice variant owns one logical macro grid, an input-phase accumulator,
and a contraction-tile accumulator. It owns no transcoder or slicer.

## Design decisions

- **No slice axes or codecs.** `program` tiles the logical matrix and transposes
  each tile from `[output_num, input_num]` to the macro contract
  `[input_num, output_num]`.
- **Value ranges delegate directly to the macro.**

## Contracts & invariants

- **Organized W shape** is
  `[..., M=1, Tc, Tr, input_num, output_num]`; **organized X shape** is
  `[..., M, Tc, Tr=1, input_num]`.
- **Aggregate** is input-phase accumulate, `Tc` accumulate, flatten
  `(Tr, output_num)`, then trim to `N`.
- **`program` shape gate.** `program(weight)` rejects any shape other than the bound `w_logical_shape`.
- **Value ranges are published, not revalidated.** The owner supplies values compatible with the published weight and activation ranges; neither transcoding nor identity slicing scans runtime tensors.

## Performance & resources

The variant adds only shape organization and exact digital accumulation.

## Gotchas

- **Last-tile padding is silent.** Trailing columns of the last output tile and trailing rows of the last contraction tile are zero-padded by `_chunk_pad_along`; the padded lanes contribute zero and are trimmed, but a caller inspecting intermediate tile shapes sees the padded extent, not `N` / `K`.

---

- **Reference**: [direct engine](../../../../../reference/architecture/unit/cim/engine/direct.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/direct.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_linear_cim_unit.py`

# `neurox/mapper/xbar/base.py`

## Current role

`XbarMapper` is the abstract mapping orchestrator that turns a high-precision logical weight / activation tensor into the xbar-primitive shape contract. Per-tile it owns:

- one tiler (geometry decomposition);
- one x slicer (activation value decomposition);
- one w slicer (weight value decomposition).

Each mapper instance is bound to one xbar tile geometry; one mapper per logical layer is the standard usage pattern.

## Why a single mapper owns both slicers

One mapper owns the entire mapping policy — tiler + both slicers — because all three share the tile-geometry and value-range invariants and must stay consistent across calls. See [`architecture/mapping.md`](../../../architecture/mapping.md).

## Mapper surface

Concrete mappers implement at least:

- `fabricate(weights)` — slice + tile the high-precision logical weight, hand the resulting xbar-native digit tensor to the xbar.
- `forward(activations)` — slice activations into per-cycle integer codes the xbar can consume.
- `to_ideal()` parity hook — let the ideal xbar twin be assembled with the same mapper.

The contract is intentionally narrow; the slicer / tiler / transcoder abstractions below carry the per-step semantics.

## Naming convention

- mapper / transcoder use `value_range`;
- xbar uses `digit_range`.

This keeps complete-value range separate from primitive-digit capability — see [`architecture/mapping.md`](../../../architecture/mapping.md).

See also:

- `simple_mapper.md`
- `slicer/README.md`
- `tiler/README.md`
- `../../../adr/ADR-0001-config-dispatch-and-owned-construction.md`

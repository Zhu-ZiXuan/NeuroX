# Mapping Architecture

This document records the current mapping split below `macro`.

## Layer split

Mapping is divided into:

- `macro`: orchestrates xbar + mapper + digital aggregation
- `mapper`: owns complete mapping semantics
- `tiler`: geometry-only decomposition
- `slicer`: value decomposition to `[..., slice_num, digit_num]`
- `transcoder`: pure digit math
- `xbar`: primitive analog VMM with primitive capabilities

## Key rules

### Macro owns one mapper

The macro does not separately own an `x_mapper` and a `w_mapper`. Instead, one mapper owns:

- one shared tiler
- one x slicer
- one w slicer

### Tiling and slicing are separate

- tiling handles matrix decomposition / padding / tile layout
- slicing handles value decomposition

They are distinct steps, but both are part of mapping, so the mapper owns both through composition.

### Slicer contract

Every slicer returns values with trailing shape:

- `[..., slice_num, digit_num]`

The generic slicer interface is driven by:

- `digit_count`
- `digit_radix`
- `digit_range`

`value_range` is a slicer output, not a slicer input.

### SimpleSlicer

`SimpleSlicer` is direct digitise-then-group:

1. encode the original integer into `slice_num * digit_count` digits
2. reshape into `[..., slice_num, digit_count]`

There is no outer / inner two-stage transcoder in the current design.

### Naming

- mapper / macro / transcoder use `value_range`
- xbar uses `digit_range`

This keeps complete-value range separate from primitive-digit capability.

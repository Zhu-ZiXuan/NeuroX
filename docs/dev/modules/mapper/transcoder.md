# `neurox/mapper/transcoder.py`

## Current role

`transcoder.py` owns the **pure-algorithm** signed-digit encoding policies. A transcoder translates an integer into a fixed-shape signed-digit list and back according to one of the supported encodings. The file is shape-agnostic and hardware-agnostic.

## Encoding policies

The current `Transcoder` family supports:

- `true_form` — straightforward base-`r` digit decomposition with signs on each digit.
- `complement` — radix-complement encoding (analogous to two's complement for `r = 2`).
- `canonical` — non-adjacent-form-style canonical signed-digit encoding.

Each policy is a concrete `Transcoder` subclass. The encoding is selected by name through the string carried in `SignedDigitTranscoder`.

## What transcoders are not responsible for

- Tile-level layout, padding, or geometry.
- Per-value digit-count / digit-radix decisions — these arrive as arguments.
- Reference-column placement.
- Analog-domain digit weighting.

This is the smallest possible abstraction: take an integer plus a digit schema, produce the corresponding signed-digit list (and vice versa).

## Why `value_range` is an output

The transcoder reports `value_range` based on its `(digit_count, digit_radix, encoding)` triple — it is an output, not an input.

See also:

- `xbar/slicer/README.md`
- `xbar/base.md`
- `../../architecture/mapping.md`

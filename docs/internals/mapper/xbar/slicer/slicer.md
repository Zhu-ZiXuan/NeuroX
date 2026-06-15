# Slicer — Implementation

## Summary

The slicer layer is the `Slicer` ABC (`base.py`) plus two concrete decompositions: `SerialSlicer` (`serial.py`, the activation path) and `SimpleSlicer` (`simple.py`, the weight path). Each wraps a [transcoder](../../transcoder/README.md) and regroups its output into the slice/digit lattice. Spec: [reference/mapper/xbar/slicer](../../../../reference/mapper/xbar/slicer/README.md).

## Design decisions

- **The ABC is a pure interface - no shared state, no `RegistryMixin`.** A slicer is selected by the data role at its construction site, not by a runtime string discriminator, so there is no registry here (unlike the transcoder). The base declares only the observable surface and leaves all state to the subclass.
- **Each subclass takes only the constructor parameters it actually uses.** `SerialSlicer` takes `slice_num` and `digit_radix`; `SimpleSlicer` adds `digit_count` and `encoding`. The slicer does not accept and then re-validate parameters a caller would have to fabricate - structural defaults (`SerialSlicer`'s `digit_count == 1`, its unsigned true-form grid) are internal. This is why `SerialSlicer` takes no `encoding`: its grid is unsigned, so the encoding is fixed and a parameter would be a lie.
- **One transcoder covers the whole digit string; the slicer only regroups.** `SimpleSlicer` builds a single transcoder over `slice_num * digit_count` digits and unflattens the trailing axis, rather than running a per-slice transcoder. The encode is one pass; the slice structure is a reshape.
- **The construction inputs are not re-exposed.** Only `value_range`, `slice_radix`, `slice_weights`, and `slice()` are observable; no caller reads `slice_num` / `digit_count` back through the slicer, so they stay private. `slice_weights` is part of the contract even when a given consumer does not read it, because it names how the slice axis recombines.

## Contracts & invariants

- **ABC observable surface.** A `Slicer` exposes exactly: `slice(x) -> Tensor` whose return carries trailing-2 axes `[slice_num, digit_num]`; and the properties `value_range: tuple[int, int]`, `slice_radix: int`, `slice_weights: tuple[int, ...]` (LSB-first $(1, R, R^2, \dots)$ with $R$ = `slice_radix`). All three are `@property` because they are init-determined constants.
- **Output shape contract.** `SerialSlicer.slice(x).shape == x.shape + (slice_num, 1)` (structural singleton digit axis); `SimpleSlicer.slice(x).shape == x.shape + (slice_num, digit_count)`. The slice axis is least-significant-first, matching `slice_weights`.
- **`slice_weights` is a tuple of Python ints, materialised at the consumer's boundary.** It is not a tensor; a consumer builds `torch.tensor(slicer.slice_weights, dtype=..., device=...)` at its own boundary. Keeping it a plain tuple avoids the slicer owning a device or dtype.
- **Digit-grid compatibility is the caller's invariant.** The slicer does not check that its `[slice_num, digit_num]` lattice matches a particular cell grid; the construction site (which forwards the cell's digit count, digit radix, and input grid into the right subclass) owns that match.

## Performance & resources

- The decomposition is a single transcoder encode plus a reshape (`unsqueeze` / `unflatten`); integer-elementwise and negligible against the analog solve. No chunking or memory model at this layer.

## Gotchas

- **`value_range` is the algorithm-side complete-value range, not the primitive single-cell range.** The xbar's per-cell range is `digit_range`; conflating the two is the central naming pitfall (see [reference value_range vs digit_range](../../../../reference/mapper/xbar/slicer/slicer.md#value_range-vs-digit_range)). A slicer never re-exposes the cell-level digit range.
- **Do not pass a signed `encoding` expecting the serial path to honour it.** `SerialSlicer` is hard-wired to true-form and takes no `encoding`; the activation primitive cell is unsigned and a signed alphabet would emit negative digits it cannot carry.
- **Out-of-range inputs wrap silently.** The value range is inherited from the underlying transcoder and is a caller contract, not a clamp; an input above the band wraps through the encode.

## Known limitations

- N/A.

---

- **Reference**: [slice decomposition](../../../../reference/mapper/xbar/slicer/slicer.md)
- **Implementation**: `neurox/mapper/xbar/slicer/base.py`, `neurox/mapper/xbar/slicer/serial.py`, `neurox/mapper/xbar/slicer/simple.py`
- **Tests**: `tests/test_slicer.py`
- **Decisions**: N/A — no ADR governs this module.

# Slicer base

`Slicer` defines the value-decomposition surface shared by the slicer family.

## Design decisions

- **The ABC is a pure interface — no shared state, no `RegistryMixin`.** A slicer is selected by the data role at its construction site, not by a runtime string discriminator, so there is no registry here (unlike the transcoder). The base declares only the observable surface and leaves all state to the subclass.
- **The construction inputs are not re-exposed.** Only `value_range`, `slice_radix`, `slice_weights`, and `slice()` are observable; no caller reads `slice_num` / `digit_count` back through the slicer. `slice_weights` is part of the contract even when a given consumer does not read it, because it names how the slice axis recombines.

## Contracts & invariants

- **ABC observable surface.** A `Slicer` exposes exactly: `slice(x) -> Tensor` whose return carries trailing-2 axes `[slice_num, digit_count]`; and the properties `value_range: tuple[int, int]`, `slice_radix: int`, `slice_weights: tuple[int, ...]` (LSB-first $(1, R, R^2, \dots)$ with $R$ = `slice_radix`). All three are `@property` because they are init-determined constants.
- **`slice_weights` is a tuple of Python ints, materialised at the consumer's boundary.** It is not a tensor; a consumer builds `torch.tensor(slicer.slice_weights, dtype=..., device=...)` at its own boundary. Keeping it a plain tuple avoids the slicer owning a device or dtype.
- **Digit-grid compatibility is external to the interface.** The base does not check that a returned `[slice_num, digit_count]` lattice matches a particular execution grid.

## Gotchas

- **`value_range` is the algorithm-side complete-value range, not the primitive single-cell range.** The xbar's per-cell range is `digit_range`; conflating the two is the central naming pitfall (see [value_range vs digit_range](../../../../../conventions/glossary.md#value-domain-and-slicing)). A slicer never re-exposes the cell-level digit range.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/base.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`

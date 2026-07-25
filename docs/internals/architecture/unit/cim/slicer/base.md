# Slicer base

`Slicer` defines the value-decomposition surface shared by the slicer family.

## Design decisions

- **The ABC is a pure interface — no shared state and no registry.** A slicer is selected by the data role at its construction site. The base declares only the observable surface and leaves all state to the subclass.
- **The construction inputs are not re-exposed.** Only `value_range`, `slice_radix`, `slice_weights`, and `slice()` are observable. `slice_weights` names how the slice axis recombines.

## Contracts & invariants

- **ABC observable surface.** `slice(x)` returns a tensor with trailing
  `[slice_num]`; `value_range`, `slice_radix`, and `slice_weights` describe the
  complete input domain and its LSB-first reconstruction.
- **`slice_weights` is a tuple of Python ints, materialised at the consumer's boundary.** It is not a tensor; a consumer builds `torch.tensor(slicer.slice_weights, dtype=..., device=...)` at its own boundary. Keeping it a plain tuple avoids the slicer owning a device or dtype.
- **Macro compatibility is external to the interface.** The consumer chooses a
  slicer whose output values fit the target macro's logical value range.

## Gotchas

- **`value_range` is the complete algorithm-side range.** It is distinct from
  the macro range carried by one output slice.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/base.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`

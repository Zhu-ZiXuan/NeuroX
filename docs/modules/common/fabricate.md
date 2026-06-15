# `neurox/common/mixin/fabricate.py`

## Current role

`FabricateMixin` provides the automatic pre-order `fabricate()` cascade used by every layered module that participates in static-mismatch sampling. Concrete classes inherit it alongside `nn.Module` and override only the per-layer sampling step.

## Contract

- Host class must inherit `nn.Module` (for `self.children()`).
- Host's `__init__` sets `self._inst_shape: tuple[int, ...]` — the per-instance multiplicity at this layer.
- Subclasses override `_sample_fabricate_mismatch(self) -> None` to resample owned static state. Default is a no-op for cascade-only container nodes that hold no static state of their own.
- `fabricate()` is auto-implemented: it calls `_sample_fabricate_mismatch()` then recurses to each `FabricateMixin` child in pre-order.

## Cascade

`_fabricable_children()` yields:

- direct children that are `FabricateMixin`;
- items of an `nn.ModuleList` or `nn.ModuleDict` child that are `FabricateMixin`;
- nothing else.

Non-FabricateMixin children are silently skipped. This lets auxiliary `nn.Module` attributes (helpers, parameters wrapped in `nn.Parameter`, etc.) coexist without participating in fabrication.

## Idempotency

`fabricate()` may be re-called any number of times. Each call resamples from the host's nominal buffers; no state accumulates. See [`docs/dev/architecture/fabrication_lifecycle.md`](../../dev/architecture/fabrication_lifecycle.md).

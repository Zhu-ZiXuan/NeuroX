# XbarMacro base

## Summary

The `XbarMacro` registry root (`xbar/base.py`): the abstract `FabricateMixin + nn.Module + ProfileMixin + RegistryMixin` orchestration node, its empty-field config/policy roots, the xbar-build helper, and the one shared geometric primitive. It declares signatures only; each mode writes its `program` / `matmul` end to end.

## Design decisions

- **Registry dispatch keyed on config type.** `from_config` looks up the impl registered for `type(config)` via `RegistryMixin`; adding a mode is one `@XbarMacro.register_key(MyConfig)` line and never touches the base. The cost — one config dataclass per mode — is paid once.
- **No template method for the run path.** The base owns construction scaffolding only (`__init__` context, `_build_xbar`, `chunk_pad_along`); it provides no shared `program` / `matmul` skeleton. The reason is the organize/aggregate duality: a mode that keeps `Sw` outside the data axis aggregates by cross-tile shift-add, while a mode that folds `Sw` into the column axis aggregates by intra-tile stride-`Sw` shift-add. The two reductions share no common reduce-all-slices helper, so a template method would only fit one mode and force the other to override it. Each mode owns its paired organize + aggregate.
- **`Sw` layout vs `Sa` schedule split.** `Sw` (per-weight slice count) is burned in at `program` time — it decides how slices occupy physical cells and is fixed until the next `program`. `Sa` (per-activation slice count) is evaluated per `matmul` call — it is a serial-cycle schedule the simulator currently materializes as a tensor axis and batches. This is why `program` and `matmul` cannot share a uniform slice-reduction helper, and why the organize step (which defines a mode) lives on the W side only.
- **The base owns no `xbar`.** `XbarMacro` does not declare an `xbar` attribute, because the degenerate ideal member has none. xbar-using modes declare `xbar: Xbar` themselves and build it via `_build_xbar`. The base never reads a mode's `xbar_config` — it cannot assume the concrete config carries one, so `_build_xbar` takes `xbar_config` and `xbar_policy` as explicit arguments.
- **Not a `CircuitBase`.** The family is an orchestration node, not silicon; its PPA aggregates through the children. See [base](../base.md) for the family-wide rationale.
- **No self-compiled forward.** The base adds no compute. Each mode's `matmul` is wrapped in `@torch.no_grad()` and runs eager — the library does not self-compile the forward ([compile](../../compile/README.md)). The memory-sensitive cost is the tile read inside `self.xbar.vec_mat_mul`, whose heavy DC solve compiles as a separate regional leaf — see [xbar internals](../../xbar/README.md).

## Contracts & invariants

- **Family-wide construction signature.** Every member's `__init__` / `from_config` is kwarg-only `(*, config, policy, name, w_logical_shape, dtype, T__K, ideal_xbar)` with no defaults (physical-layer no-defaults rule). `w_logical_shape` is `(*prefix, N, K)`; `dtype` / `T__K` propagate to every analog/digital child; `ideal_xbar` is honoured by xbar-using modes (swaps the tile for its ideal twin) and ignored by the degenerate member. Stochastic-vs-deterministic rounding is governed by `self.training` at the consuming quantiser, never a constructor override.
- **Config-field read discipline.** Before the tile exists, a mode reads only `col_num` / `row_num` off the base `XbarConfig`; everything else (`w_digit_count`, `w_digit_radix`, `x_range`, `w_digit_range`) is read off the constructed `self.xbar` via its abstract properties. A mode must not reach into xbar-subclass-specific config fields.
- **`_build_xbar` honours `ideal_xbar`.** When `ideal_xbar=True` the helper applies `.to_ideal()` and discards the passed `xbar_policy` in favour of an empty `IdealXbarPolicy`. The derived `inst_shape` is the per-instance multiplicity prefix; the xbar appends its own trailing `(col_num, w_digit_count, row_num)`.
- **`chunk_pad_along` is the one shared geometric primitive.** It right-pads an axis to a multiple of `chunk_size` then unflattens it into `(num_chunks, chunk_size)`, inserting the chunk axis immediately after. `pad_value` has no default. Every mode's tiling step funnels through it so the pad/unflatten convention is identical across modes.
- **Empty config/policy roots.** `XbarMacroConfig` has no fields (it is only the registry key) and `XbarMacroPolicy` is an empty marker; a mode declares its own `*Config` (with `xbar_config` plus slice/reducer fields) and `*Policy` (with an `xbar: XbarPolicy` field).
- **Fabrication cascades through children.** `FabricateMixin` auto-resamples the owned tile and reducers; the macro registers no static state of its own.
- **The `[Sa, Sw, Tc, Tr]` order is load-bearing for the aggregate.** The aggregate reductions address their axes by negative index against this fixed order; a mode that reorders or inserts an axis without matching the canonical layout silently reduces the wrong dimension. Absent axes are omitted, not size-1 padded, so a negative index that assumes a missing axis is present is wrong — each mode's aggregate index set is mode-specific.
- **Organize and aggregate are a matched pair.** Do not mix one mode's organize with another's aggregate; the duality (cross-plane vs intra-tile `Sw` reduction) means a mismatched pair produces a wrong-but-plausible shape. There is no generic aggregation extension point on the base by design.

---

- **Reference**: [reference/macro/xbar](../../../reference/macro/xbar/README.md)
- **Implementation**: `neurox/macro/xbar/base.py`
- **Tests**: `tests/test_xbar_macro.py`

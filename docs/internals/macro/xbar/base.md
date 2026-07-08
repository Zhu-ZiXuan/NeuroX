# XbarMacro base

The `XbarMacro` registry root: a `FabricateMixin + nn.Module + ProfileMixin + RegistryMixin` orchestration node holding the empty-field config/policy roots, the `_build_xbar` helper, and the `chunk_pad_along` primitive. It declares the `program` / `matmul` signatures only; each mode writes them end to end.

## Design decisions

- **Registry dispatch keyed on config type.** `from_config` looks up the impl registered for `type(config)` via `RegistryMixin`; adding a mode is one `@XbarMacro.register_key(MyConfig)` line and never touches the base. The cost — one config dataclass per mode — is paid once.
- **No template method for the run path.** The base owns construction scaffolding only (`__init__` context, `_build_xbar`, `chunk_pad_along`); it declares `program` / `matmul` abstract and provides no shared run-path skeleton and no aggregation extension point. A mode's organize and aggregate form a matched pair whose reductions differ per mode (the organize/aggregate dual), so they share no common reduce-all-slices helper: a template method would fit one mode and force the other to override it. Each mode owns its paired organize + aggregate, and mixing one mode's organize with another's aggregate produces a wrong-but-plausible shape.
- **`Sw` layout vs `Sa` schedule split.** `Sw` (per-weight slice count) is fixed at `program` time and held until the next `program`; `Sa` (per-activation slice count) is evaluated per `matmul` call and materialized by the simulator as a batched tensor axis. Fixed at different lifecycle points, the two cannot share a uniform slice-reduction helper, and the organize step lives on the W side only.
- **The base owns no `xbar`.** `XbarMacro` does not declare an `xbar` attribute, because a degenerate member owns none. xbar-using modes declare `xbar: Xbar` themselves and build it via `_build_xbar`. The base never reads a mode's `xbar_config` — it cannot assume the concrete config carries one, so `_build_xbar` takes `xbar_config` and `xbar_policy` as explicit arguments.
- **Not a `CircuitBase`.** The family is an orchestration node, not silicon; its PPA aggregates through the children. See [base](../base.md) for the family-wide rationale.
- **No self-compiled forward.** The base adds no compute. Each mode's `matmul` is wrapped in `@torch.no_grad()` and runs eager — the library does not self-compile the forward ([compile](../../compile/README.md)). The memory-sensitive cost is the tile read inside `self.xbar.vec_mat_mul`, whose heavy DC solve compiles as a separate regional leaf — see [xbar internals](../../xbar/README.md).

## Contracts & invariants

- **Family-wide construction signature.** Every member's `__init__` / `from_config` is kwarg-only `(*, config, policy, name, w_logical_shape, dtype, T__K, ideal_xbar)` with no defaults (physical-layer no-defaults rule). `w_logical_shape` is `(*prefix, N, K)`; `dtype` / `T__K` propagate to every analog/digital child; `ideal_xbar` is honoured by xbar-using modes (swaps the tile for its ideal twin) and ignored by the degenerate member. Stochastic-vs-deterministic rounding is governed by `self.training` at the consuming quantiser, never a constructor override.
- **Config-field read discipline.** Before the tile exists, a mode reads only `col_num` / `row_num` off the base `XbarConfig`; everything else (`w_digit_count`, `w_digit_radix`, `x_range`, `w_digit_range`) is read off the constructed `self.xbar` via its abstract properties. A mode must not reach into xbar-subclass-specific config fields.
- **`_build_xbar` honours `ideal_xbar`.** When `ideal_xbar=True` the helper applies `.to_ideal()` and discards the passed `xbar_policy` in favour of an empty `IdealXbarPolicy`. The derived `inst_shape` is the per-instance multiplicity prefix; the xbar appends its own trailing `(col_num, w_digit_count, row_num)`.
- **`chunk_pad_along` is the one shared geometric primitive.** It right-pads an axis to a multiple of `chunk_size` then unflattens it into `(num_chunks, chunk_size)`, inserting the chunk axis immediately after. `pad_value` has no default. Every mode's tiling step funnels through it so the pad/unflatten convention is identical across modes.
- **Empty config/policy roots.** `XbarMacroConfig` has no fields (it is only the registry key) and `XbarMacroPolicy` is an empty marker; a mode declares its own `*Config` (with `xbar_config` plus slice/reducer fields) and `*Policy` (with an `xbar: XbarPolicy` field).
- **Fabrication cascades through children.** `FabricateMixin` auto-resamples the owned tile and reducers; the macro registers no static state of its own.
- **The `[Sa, Sw, Tc, Tr]` order is load-bearing for the aggregate.** The aggregate reductions address their axes by negative index against the fixed leading-axis order defined in [base](../base.md); a mode that reorders or inserts an axis, or assumes an omitted axis is present, silently reduces the wrong dimension, so each mode's aggregate index set is mode-specific.

---

- **Reference**: [reference/macro/xbar](../../../reference/macro/xbar/README.md)
- **Implementation**: `neurox/macro/xbar/base.py`
- **Tests**: `tests/test_xbar_macro.py`

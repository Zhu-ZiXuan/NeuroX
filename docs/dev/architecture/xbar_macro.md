# Xbar Macro Architecture

This document defines the layered structure of every xbar-backed macro and the rules each concrete mode must satisfy.

## Layered responsibilities

- **Xbar tile** — primitive analog VMM; programs / reads physical cells. Does not understand high-precision weight semantics.
- **Value-domain primitives** — cross-mode helpers that decompose / encode integer values into digit strings (slicer, transcoder).
- **Xbar macro abstract base** — declares the contract every xbar macro satisfies; carries the shared chunk-and-pad geometric utility.
- **Concrete mode subclass** — one full execution mode: owns its slicers, reducers, and the paired organize / aggregate pipeline.

## W-side pipeline (3 steps)

Every xbar macro processes a logical weight tensor through:

1. **Slice** — pure value-domain decomposition, identical across modes. Produces a trailing `[..., Sw, D]` decomposition where `Sw` is the per-weight slice count and `D` is the per-slice digit count.
2. **Organize** — mode-specific reshape / permute that decides how the `Sw` axis maps onto physical xbar layout. *This step defines the mode*: it may leave `Sw` as a separate trailing axis, merge it into the output (`N`) axis, or arrange it in any other layout.
3. **Tile** — mode-specific chunking via the base's shared chunk-and-pad helper. Operates on whatever axes the organize step has prepared.

## X-side pipeline (2 steps)

1. **Slice** — value-domain decomposition into `[..., M, K, Sa, Dx]`. The `Sa` axis represents per-cycle slices that conceptually reduce serially.
2. **Tile** — chunk `K` to match the W-side col-tile count. Insert mode-specific placeholder axes so the activation tensor broadcasts cleanly against the fabricated weight tensor.

X has no organize step.

## Organize ↔ aggregate duality

> Aggregation is the inverse of organization. The two halves are paired
> per mode and may not be mixed across modes.

If the organize step keeps `Sw` outside the data axis, aggregation reduces `Sw` across xbar slots via cross-xbar shift-add. If the organize step merges `Sw` into the col axis, aggregation reduces `Sw` within one xbar's data axis via stride-`Sw` reshape and intra-xbar shift-add.

Because of this duality, the abstract base does **not** publish a generic aggregation extension point. Each concrete mode owns its own aggregate code paired with its own organize.

## `Sw` is layout, `Sa` is schedule

- **`Sw`** (per-weight slice count) is a *layout* decision burned in at `fabricate` time. It determines how slices occupy physical cells. Every `matmul` call sees the same `Sw` layout until the next `fabricate`.
- **`Sa`** (per-activation slice count) is a *schedule* decision evaluated at each `matmul` call. It represents per-cycle serial activation slicing. The current simulator materialises `Sa` as a tensor axis and batches the cycles, but the architectural contract does not require this — a future cycle-accurate mode is free to loop over `Sa`.

This distinction explains why fabricate and matmul cannot share a uniform "reduce all slice axes the same way" helper.

## Contract surface on the abstract base

The abstract base declares signatures only; it does **not** provide a template method, and intermediate tensor shapes are subclass concerns:

- a polymorphic `from_config(cls, *, cfg, xbar, name="")` classmethod that dispatches on the concrete config type.
- abstract `w_value_range / x_value_range / output_rescale_factor` properties.
- abstract `fabricate(weight)` and `matmul(input, weight, bias, mult, rshift, zp)`.
- a static chunk-and-pad helper — the one shared geometric primitive.

A subclass writes its own `fabricate` and `matmul` end to end; the base does not orchestrate.

## Adding a new mode

1. Define a concrete config dataclass extending the base config, declaring all sub-module configs the mode needs (slicer params, reducer configs, …).
2. Define a concrete macro subclass and register it against the concrete config with the family registry decorator.
3. Build sub-modules in `__init__` from the config; store the xbar handle and the slicers / reducers as `nn.Module` children.
4. Implement the mode-specific organize for W and X as private methods of the subclass.
5. Implement `fabricate(weight)` to call organize → `xbar.fabricate` and pre-warm reducer shapes.
6. Implement `matmul(...)` to organize → primitive VMM → aggregate (the dual of organize) → requantize.
7. Add full shape annotations on every reshape / permute step, marking placeholder axes as `=1`.

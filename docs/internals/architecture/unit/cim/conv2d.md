# Conv2dCimUnit

The engine-backed CIM unit exposing the conv2d operator through the Toeplitz / input-stationary lowering: `Conv2dUnit` × `EngineBackedCimUnit`. The mapping spec — geometry formulas, placement law, window-group rule — is [reference/unit/conv2d](../../../../reference/architecture/unit/conv2d.md).

## Design decisions

- **Toeplitz weight, strip activation — not digital im2col.** `_weight_to_matrix` unrolls the 4-D kernel into a sparse `(N', K') = (W_g*C_out, C_in*kh*W_strip)` Toeplitz matrix at program time; `_conv2d_planes` gathers `W_strip`-wide input strips (a light gather, not a full im2col). The Toeplitz zeros do the window selecting — there are no per-window row masks anywhere. im2col is the `W_g = 1` degenerate case of the same code path.
- **`W_g` is derived at construction, never a config field.** Before the engine build (the engine is constructed on the `(N', K')` shape via `_engine_w_logical_shape`), the constructor reads `row_num` / `col_num` off `config.engine.cim_macro_config` and derives `_kw_eff`, `_w_g`, `_w_strip`, `_k_prime`, `_n_prime` as plain int attrs: the maximal `W_g` with `K' <= row_num` and `W_g*C_out <= col_num`, floored at 1.
- **The engine stays conv-ignorant.** The Toeplitz matrix is handed to `engine.program` as an ordinary logical weight, so generic slicing/tiling composes without modification — in the floored case the engine splits `K'`/`N'` across tiles as usual. Sub-phase chunking is the engine's generic mechanism; the conv unit contains no chunking logic and models no sample-and-hold (every sub-phase is independently driven).
- **Conv serial axes ride as anonymous batch.** `_conv2d_planes` keeps `[H_out, T_seg]` as the conv-introduced serial axes (`T_seg = ceil(W_out / W_g)`), entering `engine.matmul` as leading batch over `[M=T_seg, K']`; `_conv2d_fold` undoes exactly them — unflatten `N' -> (W_g, C_out)` per the column law, assemble `(H_out, T_seg*W_g)`, trim to `W_out`, arrange to trailing `[C_out, H_out, W_out]`. The trim runs before the template's bias add, so the bias lands exactly once per real output element.
- **Two zero-representability guards at construction.** The Toeplitz zeros are stored weight values, so `0 ∈ engine.w_value_range` is required unconditionally. Zero activations come from padding, strip right-fill, and last-segment surplus windows, so `padding != (0, 0)` or `W_g > 1` requires `0 ∈ engine.x_value_range`.
- **Geometry is config, kernel size is shape.** `Conv2dCimUnitConfig` adds the required `stride` / `padding` / `dilation` pairs (validated positive / non-negative); the kernel extent comes from the 4-D `w_logical_shape`, never a duplicate config field.

## Contracts & invariants

- Registered via `@CimUnit.register_key(Conv2dCimUnitConfig)`; `w_logical_shape` must be exactly `(C_out, C_in, kh, kw)` (no batch prefix).
- **Placement law.** Window `g`, out-channel `n`, kernel entry `weight[n, ci, i, j]` lands at column `c = g*C_out + n`, row `r = (ci*kh + i)*W_strip + (g*sw + j*dw)`; the strip gather's row-major flatten of `(C_in, kh, W_strip)` reproduces the same row indexing, so the contraction aligns by construction.
- `program(weight, bias=None)` shape-gates the 4-D kernel, programs the Toeplitz matrix through the engine, and programs the `(C_out,)` integer bias through the base slot.
- `conv2d` is the inherited `Conv2dUnit` template — no override; surplus windows of the last segment compute valid zero-filled results and are trimmed at fold.

## Gotchas

- **Exactness is layout-coupled.** The fold's `unflatten(-1, (W_g, C_out))` inverts the column law `c = g*C_out + n` (window outer, channel inner); swapping the order produces plausible shapes with permuted output columns. Validation is bit-exact parity against `IdealConv2dUnit` ([conv2d operator internals](../conv2d.md)).

---

- **Reference**: [conv2d mapping](../../../../reference/architecture/unit/conv2d.md), [engine family](../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/conv2d.py`
- **Tests**: `tests/architecture/unit/test_conv2d_cim_unit.py`

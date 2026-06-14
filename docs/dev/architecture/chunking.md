# Chunking strategy

The physical xbar's Newton solver runs over a broadcast tensor shaped
`[*x_batch, M, Sa, Sw, Tc, Tr, phys_col, row_num]`. The leading dims grow
quickly — at BERT-FFN scale a single full broadcast carries `~150 GB` of
node-voltage state, far past any GPU's budget. Chunking serialises the
leading positions inside `CircuitCore1T1R.cim_read` so each per-call
working set fits in memory; the profiler-aggregation contract (one
energy event + one latency event per VMM regardless of chunking) is
enforced inside the core itself.

## Per-macro organised layout

Each xbar macro emits **only the axes its design actually uses**; the
bottom layer (xbar / core / solver / readout) is rank-agnostic and
handles whichever subset shows up.

Leading order is `[Sa, Sw, Tc, Tr]` (with `M` in front and `col, D, row` trailing). Dims a macro doesn't use are absent (not padded as size-1).

| Macro | Organised inst layout | Notes |
|---|---|---|
| `DirectXbarMacro` | `[*w_batch, M=1, Tc, Tr, col, D, row]` | No `Sa`, no `Sw`, no slicing. |
| `IntraArraySliceXbarMacro` | `[*w_batch, M=1, Sa=1, Tc, Tr, col, D, row]` | Real `Sw` slices live inside `col`; no separate `Sw` axis. |
| `InterArraySliceXbarMacro` | `[*w_batch, M=1, Sa=1, Sw, Tc, Tr, col, D, row]` | One xbar plane per `Sw` slice. |

`M` and `Sa` are x-side dims; their inst-side size-1 entries are
broadcast placeholders so the x tensor's real `M` / `Sa` can be
broadcast against the inst grid. Dims that the macro doesn't use are
**simply absent** — not padded as size-1. The bottom layer accepts any
rank; `classify_leading_positions` in
`neurox/xbar/_1t1r/_chunking.py` pads with size-1 on either side as
needed before broadcast classification.

## Dim taxonomy

Each leading dim falls into one of three buckets that drive whether and
how it can be serialised.

| Class | Dims | Physical meaning | Chunk knob |
|---|---|---|---|
| **A — x-side (sequential)** | `*x_batch`, `M`, `Sa` | Independent input vectors processed serially on the same arrays | `solve_chunk_size_x` |
| **B — inst (parallel + matched)** | `Sw`, `Tc`, `Tr` | Independent physical arrays operating in parallel on the chip; matched `Tc` shares the same K-tile index on both sides | `solve_chunk_size_inst` |
| **D — atomic core** | `col_num`, `row_num` | Single physical array internal | **Never chunked** |

The two chunk sizes are orthogonal: A is the x-input batch, B is the
fab-instance enumeration, and the two compose multiplicatively. The
nested loop in `cim_read` walks **A-outer / B-inner**; mathematically
the order is irrelevant (each chunk is one independent solve) — the
nested form is chosen so the per-chunk working set is exactly
`chunk_x · chunk_inst · col · row`.

## Implementation

`Offset1T1RXbar.vec_mat_mul` is now a thin pass-through to
`core.cim_read`. The two chunk knobs
(`solve_chunk_size_x` / `solve_chunk_size_inst`) live directly on
`CircuitCore1T1RPolicy` and are read by `cim_read` itself; xbar and
macros above don't see them.

```python
@torch.compiler.disable
def vec_mat_mul(self, x, *, adc_operation_point):
    v_out_phys = self.core.cim_read(x)
    ...
```

`CircuitCore1T1R.cim_read` infers `full_shape` from the broadcast of
`rram.g__uS.shape` against `x_grid.shape` — where `x_grid = x_code.unsqueeze(-2)`
inserts a size-1 WL-fanout slot so `x_code`'s trailing `row` aligns
with g's trailing `(phys_col, row)`. The classifier then splits each
leading position into A or B from the size-1 placeholders, and either
takes the single-block path (both chunk sizes `0`) or enters the
nested chunked path. The raw `x_code` (no fanout slot) is reused
later as the WL DAC's convert input — see the per-iteration step 4.

### Chunked path

Per `(chunk_a, chunk_b)` iteration:

1. Build a flat 1-D index for the chunk's positions inside the A subset
   and inside the B subset (`torch.arange` → `torch.unravel_index`).
2. Combine the two via Cartesian-product broadcast + reshape to get
   `multi_coords`: one 1-D `(chunk_size,)` index tensor per **leading
   position** of the full broadcast.
3. Each fab-stateful device (`RRAM`, `NMOS`, `Driver`, `OpAmpTIA`)
   exposes a single `snapshot(shape, multi_coords)` method.
   The caller passes the full broadcast `shape = (*leading, *trailing)`;
   internally the device runs `state.expand(shape)` (a view) and
   `view[multi_coords]` to materialise exactly the chunk's slice.
   Passing `multi_coords=None` returns the broadcast view directly
   (the tool / test path). The noise sources (telegraph / Gaussian /
   Pelgrom) are sampled at the chunk size, preserving per-instance
   independence.
4. `wl_dac.convert` runs **once outside the loop** on the zero-copy
   `x_code.expand(*leading, row)` broadcast view. The DAC sees its
   natural `(*leading, row)` shape (no WL-fanout slot, which would be
   a synthetic singleton dim unrelated to the DAC's structure); the
   fanout is re-added via `unsqueeze(-2)` *after* convert so the
   solver sees the expected `(*leading, 1, row)` view. The DAC output
   materialises at `(*leading, row)` — small relative to the per-chunk
   Newton working set — and converting against the realised full
   leading is required so per-instance dynamic energy is counted once
   per instance and `drive_thermal__V` samples an independent noise
   pattern per instance. Each VMM logs exactly one DAC energy +
   latency event pair. The per-chunk WL voltage is an advanced-index
   lookup on the full DAC output.
5. The chunk goes through `solver.solve_dc` and `tia.solve_dc`. The
   chunk's per-position energy is computed via
   `_compute_array_energy__fJ(solver_dcop=..., v_wl_drive=...)` and
   accumulated. The heavy `solver_dcop_chunk` (carrying
   `v_bl_node` / `v_sl_node` / `v_x_node` / `i_cell` sized
   `(chunk_size, phys_col, row)`) goes out of scope at iteration end —
   Python's GC frees it before the next chunk allocates. **Only
   `v_out_phys` + the chunk energy scalar are appended to per-field
   lists**; nothing else survives the loop.

After all chunks: `v_out_phys` is scatter-reassembled into
`(*leading, phys_col)`, per-chunk energies are reassembled into
`(*leading,)`, then one `_log_dynamic_energy` + one `_log_latency`
event are emitted for this VMM (inline at the end of `cim_read`, the
plain-forward pattern shared with every other emitting leaf). The
function returns `v_out_phys`. There is no `Core1T1RDCOP` — the core
has no DCOP of its own; the sub-solvers' DCOPs are private to the
loop and used in place for energy aggregation only.

The same unified loop handles both "single block" and "chunked":
when both `solve_chunk_size_x` and `solve_chunk_size_inst` are `0`,
`iter_chunks` yields exactly one chunk spanning the whole broadcast
leading. There is no parallel non-chunked code path.

Inner-solver debug (residuals, per-cell state) goes through standalone
solver construction — see `circuit_core.md` "Driving the inner solver
standalone for debug / calibration".

### Memory accounting

Per-chunk **solver** working set: `chunk_x · chunk_inst · col · row`
elements per node-voltage field × 4 fields × dtype. Independent of
total batch / model size — this is the dominant allocation chunking
bounds. The reassembly buffer for `v_out_phys` is `(B_total, col)` —
small relative to the per-chunk solver state.

Not bounded by chunking: the WL DAC output is materialised once,
outside the loop, at shape `(*leading, row)` (then `unsqueeze(-2)` for
the solver). It is **not** chunked. This trade-off keeps:

- exactly **one** DAC energy + latency event per VMM (chunking would
  emit one event per chunk, breaking the per-VMM event invariant);
- per-instance independent `drive_thermal__V` noise (chunking would
  sample noise at chunk-shape, making the RNG stream chunk-size
  dependent — and degenerate to "same noise across inst" if the chunk
  collapses inst).

For BERT-FFN-scale leading the DAC output is ~16 MB fp64 — still about
an order of magnitude smaller than a single un-chunked solver working
set, so chunking still saves substantial peak memory. The DAC buffer
just isn't covered by the `chunk_x · chunk_inst` knobs.

## Bit-exactness scope

Bit-exactness against the single-block path holds **only under
deterministic (noise-off) policies**. With any stochastic source
enabled (`*_mismatch` flags, `read_thermal`, `read_telegraph`,
`comparator_*_noise`, `mux_noise_*`, `drive_thermal`, …) the per-chunk
``snapshot(...)`` calls consume the device's RNG stream at chunk-shape
granularity instead of full-leading granularity — different chunk
sizes change the order and amount of random draws, so the output
tensors differ element-wise even though they remain statistically
equivalent (same distribution, same variance). Pin a single chunk
size when bit-exact comparison against a noisy reference is required.

## Test coverage

- `tests/test_xbar_chunking.py` (integration, noise-off policy)
  - A-axis bit-exact across chunk sizes 1, 2, 4, 8.
  - B-axis bit-exact across chunk sizes 1, 2, 4.
  - Mixed `(A, B)` bit-exact for representative pairings.
  - Remainder chunks on both axes.
  - Profiler emits exactly **one** core energy event + **one** core
    latency event per VMM under every chunking configuration, with
    total energy bit-exact versus the single-block path.
- `tests/test_chunking_helpers.py` (unit-level)
  - `iter_chunks` `multi_coords` / `flat_global_idx` ordering on known
    `(A, B)` layouts.
  - `reassemble_chunks` scatter correctness, including the
    `leading=()` short-circuit.
  - `_row_major_strides` on representative shapes.
  - Multi-leading-dim `vec_mat_mul` integration with leading shaped
    like a real macro (`(M, Sa, Sw, Tc, Tr)`).

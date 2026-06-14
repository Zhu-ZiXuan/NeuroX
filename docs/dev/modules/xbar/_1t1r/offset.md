# `neurox/xbar/_1t1r/offset.py`

`Offset1T1RXbar` is the current offset-coded xbar implementation.

It owns:

- physical-shape regrouping and offset/reference-column semantics;
- one `CircuitCore1T1R`;
- one `ReadOut`;
- the xbar-level capability surface defined by the generic `Xbar` base.

`Offset1T1RXbarPolicy(XbarPolicy)` is a structured composite policy with one sub-policy per child:

- `core: CircuitCore1T1RPolicy` — also carries the chunk knobs (`solve_chunk_size_x` / `solve_chunk_size_inst`). They live on the core's policy because that is where the chunked solve actually runs; values depend on the host environment (GPU memory budget, target throughput, concurrent jobs) rather than the physical chip preset, so the same TOML can be reused across hosts with chunk sizes picked per-host.
- `readout: ReadOutPolicy` — abstract base; the concrete impl (e.g. `OffsetSwitchCapMuxAdcReadOutPolicy`) is passed by the caller.

`vec_mat_mul(x, *, adc_operation_point)` is a thin pass-through (decorated with `@torch.compiler.disable` to keep upstream macro-level `@torch.compile` from tracing into the inner numeric block, which would otherwise hit `mcs_sar.py` graph breaks). It calls `core.cim_read(x)` to obtain `v_out_phys` and runs the readout chain on that. All chunk bookkeeping — the nested A-outer / B-inner loop, snapshot slicing, per-chunk solver / TIA / energy, and final scatter-reassembly — lives in `CircuitCore1T1R.cim_read`, which reads the chunk knobs directly off its own `policy`. See `docs/dev/architecture/chunking.md`.

`CircuitCore1T1RPolicy.solve_chunk_size_x` chunks the **A subset** of the broadcast leading (x-side positions `*x_batch`, `M`, `Sa`); `solve_chunk_size_inst` chunks the **B subset** (inst positions `Sw`, `Tr`, matched `Tc`). Both are `0` by default (no chunking → single-block path). Either non-zero forces the core into the chunked path. Bit-exactness against the single-block path holds **only under deterministic (noise-off) policies**; see `docs/dev/architecture/chunking.md` for the RNG-stream caveat, and `tests/test_xbar_chunking.py` for the deterministic regression suite.

Inner `@torch.compile` on the readout block was attempted but produced > 10 min compile times dominated by inductor scheduling of the SAR ADC's bit-loop. Reintroducing block-level compile requires rewriting the SAR ADC to a graph-friendly form first.

Current construction rule:

- `Offset1T1RXbar.__init__(*, config, policy, name, inst_shape, dtype, T__K)` receives `inst_shape` (the per-instance multiplicity prefix). The full digit-tensor shape `self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)` is derived inside the xbar.
- The xbar derives its children's shapes once at `__init__`:
  - `core.w_layout_shape = (*inst_shape, physical_col_num, row_num)` where `physical_col_num = col_num * w_digit_count + n_groups`.
  - `readout.inst_shape = (*inst_shape, n_groups)`.
- Family dispatch happens inside the respective family bases.

`Offset1T1RXbar.program(w)` performs the logic-to-physical column scatter (data-major / digit-minor, plus reference-column insertion) and the offset shift, then calls `core.program(w_state_idx)`. `fabricate()` is the inherited auto-cascade; the offset xbar owns no static mismatch itself.

The offset xbar is the place where logical layout becomes physical layout. The core below it remains encoding-agnostic.

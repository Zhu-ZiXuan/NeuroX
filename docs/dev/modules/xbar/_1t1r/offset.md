# `neurox/xbar/_1t1r/offset.py`

`Offset1T1RXbar` is the current offset-coded xbar implementation.

It owns:

- physical-shape regrouping and offset/reference-column semantics;
- one `CircuitCore1T1R`;
- one `ReadOut`;
- the xbar-level capability surface defined by the generic `Xbar` base.

`Offset1T1RXbarPolicy(XbarPolicy)` is a structured composite policy with one sub-policy per child:

- `core: CircuitCore1T1RPolicy`
- `readout: ReadOutPolicy` — abstract base; the concrete impl (e.g. `OffsetSwitchCapMuxAdcReadOutPolicy`) is passed by the caller.
- `execution: ExecutionPolicy` — runtime knobs that are not physical chip parameters. Currently exposes a single field `batch_chunk_size: int`. Values `<= 0` disable chunking; positive values bound peak per-VMM memory by chunking the leading batch axis of `x` and looping the inner Newton + readout block.

The xbar forwards each sub-policy verbatim into the matching child.

`vec_mat_mul(x, *, adc_operation_point)` is the chunk scheduler (decorated with `@torch.compiler.disable` to keep upstream macro-level `@torch.compile` from tracing into the inner numeric block, which would otherwise hit `mcs_sar.py` graph breaks). The actual per-block solve is `_vec_mat_mul_block`: it runs one `core.solve_dc` plus the readout chain, returning only the per-chunk ADC code tensor. All row-shape intermediates (Newton-solver internal state, switch-cap voltages, `Core1T1RDCOP` fields) stay local and are eligible for garbage-collection between chunks. Chunking is bit-exact under deterministic policies — see `tests/test_xbar_chunking.py`.

Inner `@torch.compile` on `_vec_mat_mul_block` was attempted but produced > 10 min compile times dominated by inductor scheduling of the SAR ADC's bit-loop. Reintroducing block-level compile requires rewriting the SAR ADC to a graph-friendly form first.

Current construction rule:

- `Offset1T1RXbar.__init__(*, config, policy, name, inst_shape, dtype, T__K)` receives `inst_shape` (the per-instance multiplicity prefix). The full digit-tensor shape `self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)` is derived inside the xbar.
- The xbar derives its children's shapes once at `__init__`:
  - `core.w_layout_shape = (*inst_shape, physical_col_num, row_num)` where `physical_col_num = col_num * w_digit_count + n_groups`.
  - `readout.inst_shape = (*inst_shape, n_groups)`.
- Family dispatch happens inside the respective family bases.

`Offset1T1RXbar.program(w)` performs the logic-to-physical column scatter (data-major / digit-minor, plus reference-column insertion) and the offset shift, then calls `core.program(w_state_idx)`. `fabricate()` is the inherited auto-cascade; the offset xbar owns no static mismatch itself.

The offset xbar is the place where logical layout becomes physical layout. The core below it remains encoding-agnostic.

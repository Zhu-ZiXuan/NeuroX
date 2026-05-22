# `neurox/xbar/_1t1r/offset.py`

`Offset1T1RXbar` is the current offset-coded xbar implementation.

It owns:

- physical-shape regrouping and offset/reference-column semantics;
- one `CircuitCore1T1R`;
- one `ReadOut`;
- the xbar-level capability surface defined by the generic `Xbar` base.

Current construction rule:

- `Offset1T1RXbar.__init__(*, cfg, name, w_layout_shape, dtype, T__K)` receives `w_layout_shape = (*prefix, col_num, w_digit_count, row_num)`.
- The xbar derives its children's shapes once at `__init__`:
  - `core.w_layout_shape = (*prefix, physical_col_num, row_num)` where `physical_col_num = col_num * w_digit_count + n_groups`.
  - `readout.inst_shape = (*prefix, n_groups)`.
- Family dispatch happens inside the respective family bases.

`Offset1T1RXbar.program(w)` performs the logic-to-physical column scatter (data-major / digit-minor, plus reference-column insertion) and the offset shift, then calls `core.program(w_state_idx)`. `fabricate()` is the inherited auto-cascade; the offset xbar owns no static mismatch itself.

The offset xbar is the place where logical layout becomes physical layout. The core below it remains encoding-agnostic.

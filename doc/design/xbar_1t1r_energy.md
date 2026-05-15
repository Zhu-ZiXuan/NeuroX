# 1T1R Crossbar Dynamic Energy — Design

This document describes how the 1T1R xbar reports per-VMM dynamic
energy after the [three-layer refactor](core_1t1r.md): the array
boundary (`E_array` + `E_sl_driver`) lives inside `Core1T1R`
([`neurox/xbar/core_1t1r.py`](../../neurox/xbar/core_1t1r.py)); the
mapping layer (`Offset1T1RXbar`) adds the readout / ADC contributions
on top.  It is a thin companion to the solver design in
[`solver_1t1r.md`](solver_1t1r.md) and serves as the single source of
truth for the energy formulas the code implements.

## 1. Scope

Every call to `Offset1T1RXbar._vec_mat_mul_impl` returns a 0-d energy
tensor summed across:

```
dynamic_energy__fJ = core.dynamic_energy           # E_array + E_sl_driver
                   + readout.dynamic_energy        # per-data sum reduced to per-VMM
                                                   #   = data_sc + ref_sc + mux + adc
                                                   #     + orchestrator share
```

After the readout-bundle refactor (`temp/readout.md`), the readout
owns the data/ref SwitchCap S/H, the AnalogMux differential
transport, and the differential ADC.  `ReadOutOutput.dynamic_energy__fJ`
is shape `(*R, G, M)` (the grouped readout lattice — `G` ref groups,
`M` data per group) and is the per-data sum of each sub-block's
contribution; the xbar reduces along the last two dims to obtain
the per-VMM total.  See `signal_chain.md` §"Per-VMM energy
bookkeeping".

The peripheral terms (each sub-block's `energy_per_op__fJ` table)
are described in their respective module docs.  This doc covers
only `E_array`, which lives inside `Core1T1R._compute_array_energy__fJ`
and is decomposed as follows:

```
E_array = E_thermal                         # resistive (WL-high window)
        + E_WL_drive_ground                 # C at WL seen as pure ground cap
        + E_BL_recover_ground (at Node X)   # C_db + c_bot, active cells
        + E_BL_recover_ground (at BL nodes) # wire + c_top, all nodes
        + E_Cgd_Miller                      # inter-node C_gd, Miller-reduced
```

## 2. NMOS cap model — four caps + Miller

All four NMOS parasitic caps come from the physical
:class:`neurox.device.nmos.NMOS` model (see
[`nmos_model.md`](nmos_model.md)) — compiled from geometry and PDK
cap densities at construction time, not extracted macro fF values.
The NMOS drain (Node X) and source (SL) sit at different potentials
in the steady state, so their gate couplings differ and must be
tracked separately.  The bulk is tied to ground:

| Cap | Between | Physical source (NMOS) |
|---|---|---|
| `c_gs` | Gate (WL) ↔ Source (SL) | ``½·W·L·C_ox + W·C_gso`` |
| `c_gd` | Gate (WL) ↔ Drain (Node X) | ``½·W·L·C_ox + W·C_gdo`` |
| `c_db` | Drain (Node X) ↔ Ground | ``A_D·C_j + P_D·C_jsw`` |
| `c_sb` | Source (SL) ↔ Ground | ``A_S·C_j + P_S·C_jsw`` |

Read-cycle voltage swings at each cell (state 0 → state 1):

```
ΔV_WL =  V_DD,WL                  (large, positive — driven by the WL DAC)
ΔV_X  = -(bl_drive − V_X^(1))     (negative — Node X dips during steady state)
ΔV_SL ≈  0                         (SL clamped near ground)
```

`bl_drive` is the **dynamic** BL clamp voltage per physical column —
produced by the non-linear TIA inside the solver and exposed on
``SolverResult.bl_drive`` (shape ``[*batch, phys_col_num]``).  Under
heavy column draw the op-amp's finite gain drags it noticeably
below the TIA's static ``v_ref``, so the energy expressions broadcast
a per-column tensor through the formulas below rather than a
tile-wide scalar.

Applying **Miller's theorem** (a cap `C` between nodes A and B with
voltage swings `ΔV_A`, `ΔV_B` is equivalent, in supply-energy terms,
to independent ground caps with coefficients `(1 − ΔV_B/ΔV_A)` at A
and `(1 − ΔV_A/ΔV_B)` at B):

* **`c_gs`**: `ΔV_SL ≈ 0` ⇒ Miller factor at WL = 1 ⇒ `c_gs` is a pure
  ground cap at the WL node.  No capacitive energy at SL (SL supply is
  at 0 V, so `V_supply · ΔQ = 0` regardless).
* **`c_gd`**: both `ΔV_WL` and `ΔV_X` are non-trivial and of opposite
  sign — Miller enlarges the effective ground cap at both nodes.  The
  two Miller factors cannot be folded into a constant: they depend on
  the solver-computed `V_X^(1)` (per cell).  We carry `c_gd` through
  to the hot path and multiply it by the per-cell voltage factor
  ``V_DD,WL + bl_drive − V_X^(1)`` directly.
* **`c_db`**, **`c_sb`**: already ground caps; `c_db` enters the
  Node-X ground-cap bucket; `c_sb` contributes no supply energy
  because the SL supply is at 0 V.

## 3. RRAM cap model

Two electrode parasitics are added on `RRAMConfig`:

| Cap | Between | Role |
|---|---|---|
| `c_top` | BL wire (top electrode) ↔ Ground | Ground cap at the BL node of that cell |
| `c_bot` | Node X (bottom electrode) ↔ Ground | Ground cap at Node X |

Both are pure ground caps; no Miller reduction needed.

## 4. Wire cap model

Each `Wire` exposes `c__fF_per_um`.  The xbar integrates over the
physical path length:

```
C_wire_wl_total = c__fF_per_um · (row_first_space + (phys_col − 1) · row_cell_space)
C_wire_bl_total = c__fF_per_um · (col_first_space + (row_num  − 1) · col_cell_space)
```

`C_wire_wl_total` is per-row (all cells on that row share the WL
line).  `C_wire_bl_total` is distributed evenly across `row_num` BL
nodes — a uniform-per-node approximation that is exact for total
energy when IR drop is uniform and a few-percent approximation
otherwise (see "Caveats" below).  SL wire cap participates in
settling time but not in the supply-energy model (the SL clamp is at
0 V).

## 5. Scalar aggregates computed at `__init__`

The xbar precomputes three Python floats so the hot path carries no
device-state tensor ops:

```python
self.c_wl_per_row__fF  = C_wire_wl_total + phys_col_num * switch.c_gs__fF
self.c_bl_per_node__fF = C_wire_bl_total / row_num + rram.c_top__fF
self.c_x_per_cell__fF  = switch.c_db__fF + rram.c_bot__fF
self.c_gd_per_cell__fF = switch.c_gd__fF   # Miller-coupled, per-cell factor applied in hot path
```

`V_DD,WL` is also resolved once at `__init__` — always from the WL
DAC's on-level (`wl_dac.code_to_signal[1]` — the 2-state WL DAC's code-1
voltage).  There is **no**
config-level override: the DAC is the single source of truth for the
physical drive voltage.

## 6. Hot-path energy formulas

After the solver returns `(bl_i, sl_i, v_bl, v_x)`, every capacitive
term uses the **real** per-cell / per-row voltages from the solver —
no row-mask zeroing of "inactive" contributions.  Inactive cells
still pick up a small supply energy because their `V_X^(1)` tracks
the BL wire's IR drop (NMOS off ⇒ zero cell current ⇒ V_X = V_BL at
that node).  The per-row WL state-1 voltage `V_WL^(1) = V_DD,WL ·
wl_logic` handles the legitimate WL-driver zero-energy case for
inactive rows.

```python
V_DD_WL = self.v_dd_wl__V                               # float, resolved at __init__
v_clamp         = bl_drive__V.unsqueeze(-1)               # [*batch, 2, phys_col, 1]
v_wl_state1__V  = (V_DD_WL * wl_logic).unsqueeze(-2)      # [*batch, 2, 1, row_num]

# (1) Thermal — unchanged
array_power__mW = sum(bl_drive · bl_i) + sum(sl_drive · sl_i)
e_thermal       = array_power__mW * wl_pulse_length__ns * 1e-3

# (2) WL-side ground caps (wire + C_gs) — full CV² per row that pulses.
#     Inactive rows genuinely contribute zero (V_WL^(1) = 0 when
#     wl_logic = 0): this is physical, not a masking assumption.
e_wl_ground     = wl_logic.sum(-1) * c_wl_per_row__fF * V_DD_WL² * 1e-6

# (3) Node-X ground caps (C_db + c_bot) — every cell, real V_X^(1).
#     Active cells: large V_X drop.
#     Inactive cells: small drop tracking BL wire IR at that column.
delta_v_x__V         = v_clamp - v_x__V
e_bl_x_ground   = (v_clamp · c_x_per_cell__fF · sum_row(delta_v_x__V)).sum_col * 1e-6

# (4) BL-wire-node ground caps (wire + c_top) — every BL node.
delta_v_bl__V        = v_clamp - v_bl__V
e_bl_node       = (v_clamp · c_bl_per_node__fF · delta_v_bl__V).sum_(col,row) * 1e-6

# (5) Miller C_gd — inter-node, every cell; real per-row V_WL^(1).
#     E_Cgd_per_cell = C_gd · (V_WL^(1) + bl_drive)
#                           · (V_WL^(1) + bl_drive − V_X^(1))
sum_voltage__V  = v_wl_state1__V + v_clamp
delta_v_miller__V    = sum_voltage__V - v_x__V
e_cgd           = (sum_voltage__V · c_gd_per_cell__fF · delta_v_miller__V).sum_(col,row) * 1e-6

array_energy__nJ = e_thermal + e_wl_ground + e_bl_x_ground + e_bl_node + e_cgd
```

Unit accounting: `fF · V² = 1e-6 nJ`.  The `1e-6` is applied per
term so fp32 precision isn't spent on mixed-magnitude aggregation.

For active cells, `V_WL^(1) = V_DD,WL` and the Miller factor collapses
to the familiar `(V_DD,WL + bl_drive) · (V_DD,WL + bl_drive − V_X^(1))`.
For inactive cells, `V_WL^(1) = 0` and the factor becomes
`bl_drive · (bl_drive − V_X^(1))` — capturing the tiny BL-supply
refill energy that C_gd draws when the column's BL node experiences
IR drop.  Both cases come from the same formula with no conditional
logic.

## 7. Where the Miller terms come from (derivation sketch)

Over one 0→1→0 cycle the supplies net-deliver the following energy to
each cap, tracked by charge balance on each capacitor plate and the
corresponding supply voltage:

### `C_gs` (WL ↔ SL), SL at 0
* Charge on WL side at state 1: `C_gs · V_DD,WL`
* WL supply delivers `V_DD,WL · (C_gs · V_DD,WL) = C_gs · V_DD,WL²` during 0→1, dissipated in driver on 1→0
* Total: **`C_gs · V_DD,WL²`** at the WL supply.  SL supply: 0 (V_SL = 0).

### `C_gd` (WL ↔ X)
* ΔQ on WL side across 0→1 = `C_gd · (V_DD,WL + bl_drive − V_X^(1))`
* ΔQ on X side across 1→0 recovery = `C_gd · (V_DD,WL + bl_drive − V_X^(1))` (same magnitude)
* WL supply delivers `V_DD,WL · ΔQ` during 0→1
* BL clamp delivers `bl_drive · ΔQ` during 1→0 recovery
* Total: **`(V_DD,WL + bl_drive) · C_gd · (V_DD,WL + bl_drive − V_X^(1))`** per active cell.

### `C_db` (X ↔ ground) and `c_bot` (X ↔ ground)
* ΔQ across 1→0 recovery at X side = `(C_db + c_bot) · (bl_drive − V_X^(1))`
* BL clamp delivers `bl_drive · ΔQ` during recovery
* Total: **`bl_drive · (C_db + c_bot) · (bl_drive − V_X^(1))`** per active cell.

### `C_sb` (SL ↔ ground)
SL supply at 0 V ⇒ zero supply energy regardless of cap value.

### BL wire caps + `c_top` (BL node ↔ ground)
ΔQ during recovery per BL node = `(C_wire_bl_per_node + c_top) · (bl_drive − V_BL_node^(1))`; BL clamp delivers `bl_drive · ΔQ` per node.

## 8. Single-array multiplicity

The bias-coded `Bias1T1RXbar` runs a single physical sub-array per
logical tile — the differential ADC and reference-column subtraction
recover the signed MAC without sign-splitting.  Every capacitive
term above is evaluated once on the sole `[*batch, phys_col, row]`
broadcast; the base class's area / leakage scaling carries no
extra multiplier.

## 9. Caveats and future work

**Uniform BL-node cap approximation.**  Each BL wire has a
distributed capacitance that we lump into a scalar per-node value by
dividing the total BL wire cap evenly across `row_num` nodes.  Exact
for total energy when IR drop is uniform; a few-percent approximation
when IR drop is strongly position-dependent.  A per-node-cap-tensor
model could sharpen this but was judged not worth the complexity at
the architecture-level reporting this simulator targets.

**SL swing ignored for `C_gs`.**  We assume `ΔV_SL ≈ 0` so that
`C_gs` reduces to a pure ground cap at WL.  The solver actually
computes `v_sl` but does not return it; if a future use case
requires the small `C_gs · V_DD,WL · V_SL^(1)` correction, we can
expose `v_sl` from the solver and subtract the correction term.

**SL capacitance is stored but unused.**  `Switch.c_sb__fF` and the
SL wire cap are recorded on the device but not summed into any
energy term.  Their contribution cancels at the supply-power level
(SL clamp sits at 0 V).  They are kept on-device so a future
settling-time model can use them without re-plumbing.

**Static / leakage energy.**  This doc covers dynamic energy only.
Leakage is reported separately through `XbarMacro.leakage_energy__nJ`,
which multiplies per-instance leakage by the full `latency_per_op__ns`
(not `wl_pulse_length__ns`) because leakage is active across the
entire outer cycle.

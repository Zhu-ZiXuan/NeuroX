# Ye 2023 JSSC WH-2T1R CIM macro

`Ye2023JsscCimMacro` models one compute-in-memory macro of the 28-nm RRAM CIM chip of Ye et al. (IEEE JSSC 2023), built on the Weighted Hybrid 2T1R (WH-2T1R) cell array and the Reference-Subtracting Current Sense Amplifier (RS-CSA). The constructor binds `input_num` and `output_num` to internal `row_num` and `col_num`; the interface is then `program(w[row_num, col_num])` followed by `vec_mat_mul(x[..., row_num], *, quantization_mode, adc_bits) -> codes[..., col_num]`. Logical unsigned weights are encoded internally into the configured binary weighted planes. The scheme composes the WH-2T1R lookup cell (`Ye2023Jssc2t1rCell`), the dedicated array (`Ye2023Jssc2t1rArray`) that solves over it, and the RS-CSA readout (`RsCsaIadc`) with its dedicated reference source (`Iref`), plus the word-line and BL input converter banks, per-column BL/SL clamps, and two flat peripheral seats.

The macro config carries no geometry: the logical dimensions arrive as constructor arguments.

## Geometry

The logical contract is realized on a TRANSPOSED, digit-folded physical grid — at the paper design point 64 rows × 128 columns, laid out as four 64 × 32 sub-arrays side by side:

- **Physical rows = logical OUTPUTS** (the TBL / ladder direction): one word line per output, driven one-hot, and each row's transpose bit line driven together with it.
- **Physical columns = `row_num * (len(weight_radix) + len(redundant_radix))`**, laid out plane-major: the weight planes first (SUBA1/2/3, T2 slice multipliers LSB-first, so `weight_radix[0]` carries digit 0), then the redundant plane last (SUBA4).

The place value of each plane is ANALOG: the radix-scaled T2 currents of the selected row's cells sum on one shared transpose bitline (TBL); there is no digital shift-add of bit-planes. Encoding a logical value lands each of its binary digits on the column group of the matching place value, so the digit order stays LSB-first end to end.

`weight_radix` and `redundant_radix` split the plane list into its two roles. The redundant plane carries no weight information: `program` writes it all-HRS and `vec_mat_mul` drives its BL columns at input-0, so it contributes only its radix-weighted share of the row leakage floor. The encode lookup table and `w_value_range` therefore span `weight_radix` alone, while the leakage path spans the full radix list.

## Design point and generalization

The paper design point (32 inputs, 64 outputs, three weight planes plus one redundant plane, 1-bit inputs, and a 4-bit RS-CSA) is one config point. The array supports any positive row count, any non-empty positive-integer `weight_radix` list, and any (possibly empty) `redundant_radix` list. The macro publishes the logical envelope `w_value_range = (0, sum(weight_radix))`; an arbitrary list may leave unrepresentable holes inside that envelope, and configuration validation deliberately does not impose a dense positional-radix pattern. The design is input-parallel and output-serial: `max_active_num == row_num`, while the outputs are produced sequentially by the single time-shared RS-CSA.

## Cell and the two-step decoupled solve

The WH-2T1R cell (`Ye2023Jssc2t1rCell`) is the shared linearized 1T1R cell plus ONE new physics field. T1 (an I/O device biased half-on as a resistor at the geometric mean of the two RRAM states) in series with the RRAM forms a divider that sets the internal node `V_X`; T2 (a core device in sub-threshold saturation) converts `V_X` into the compute current on the TBL. The macro splits this into two decoupled steps, both computed every call:

1. **step1 — the BL/SL divider solve.** The shared nested DC solver (`ColBlColSlSolver`) settles the T1+RRAM divider driven by the per-column BL clamp, at the level the BL converter puts on that column, against the grounded SL, yielding the per-column BL port current and `V_X`. The BL port current is what the macro bills as the input-branch conduction. The solved `V_X` is the operating point step2 reads; no energy term reads it, and it never leaves the array.
2. **step2 — the T2 compute current, off the cell's own surface.** `I_T2` is the cell's `i_t2__uA(dcop, snap)`, a unit-slice-scale current decided by two independent branches:
    - **Is this cell's row pair driven?** The scheme drives a selected row's word line and transpose bit line TOGETHER and holds both lines of every other row at ground, so an unselected cell's T2 drain is undriven and its TBL contribution is exactly zero. That is drain-side physics, and the cell's observable for it is its own WL terminal against its calibration threshold.
    - **Which operating point does the steady state sit at?** `I_T2 = f(V_X)` is sampled at two calibration operating points, the FLOOR at `V_X = 0` and the DRIVE point at `V_X > 0`, and the solved `V_X` picks between them. At the drive point LRS gives the on-current `I_unit` and HRS the unit-scale weight leakage; the floor point gives the off-cell floor `I_0`, the per-cell contribution to the row leakage that PH0 compensates. The `0` is the table's own definition anchor, not a tunable.

    The array scales each column's unit current by its plane's place value and sums over the grid into the raw TBL current `i_tbl`, floor INCLUDED. There is **no TBL solve, no VCCS, no device model**: T2 in sub-threshold saturation is a near-ideal current source, so the current delivered to the readout equals the sum by conservation regardless of TBL wire drop.

The two operating points reach the input through a macro-level correspondence chain: an input bit selects a BL code, the BL level lands on the column, and the divider puts `V_X` on the floor point at the IN = 0 level and on the drive point at the IN = 1 level. "Input bit" is macro vocabulary throughout — the cell knows only `V_X` and its own gate.

Sub-threshold off-state current scales with device width, so the radix multiplier applies to the leakage entries as well as to the on-current. The HRS drive entry is set so that the LARGEST place value saturates the paper's measured per-plane HRS bound, which makes it a `[bound-derived]` value rather than a fit.

`Ye2023Jssc2t1rArray` is the kernel 1T1R array (`XbarArray1t1r`) plus this one extension: step1, the rail ladders, and the whole capacitive billing are the kernel's, and the scheme adds only step2's place-value sum to the per-chunk measurement and to the returned steady state. The two steps are FUSED in one call (`solve_array`): the array snapshots the cell grid at its broadcast-leading shape and issues ONE `solve_dc` through the `ChunkedSolver` middle layer, which splits the leading into `solve_chunk_size` blocks. Each block is measured down to what outlives it — both port states, the block's `i_tbl` and its array energy — before the next block is solved, so step2's current grid and the DCOP's node voltages, `V_X` included, die with their own chunk and never reach full leading. The measurement hands the cell the two things it already receives per chunk — the sliced DCOP and the sliced cell snap, which carries the per-cell WL drive as a full `[..., col, row]` grid, one value per gate — and contributes only the place values and the reduction, so no measure tensor of its own crosses the chunk loop. Both decisions inside `i_t2__uA` are the CELL's own reading of terminals it owns, its gate and its solved `V_X`, so the array carries NO threshold: the calibration threshold never leaves the cell it belongs to, and the array does not decode inputs or mask rows. The output rows are solved HONESTLY (each output its own solve, batched over the leading dimension but not folded), because each row's wire geometry differs — the column-slot fold assumption does not apply to this serial-readout dimension.

## RS-CSA readout

The Reference-Subtracting CSA (`RsCsaIadc`) is a current-domain successive-approximation ADC that is functionally a UNIFORM quantizer. It takes ONE reference current `I_ref` and scales it by its compare phases' binary weights — phase `p` (1-based, MSB-first) weighs `2**(bits - p)` — so together the phases span the ascending decision ladder `c * I_ref` for `c = 1 .. 2**bits - 1`. That ladder is INTERNAL: the tap count is the converter's own circuit fact and nothing above it states one. The RS-CSA subtracts the static PH0 compensation current and reads the code as the number of ladder taps the compensated current clears.

**The reference is injected, never held.** The macro owns a dedicated single-tap `Iref` (`reference_config`) that feeds this readout alone. It carries one tap per declared quantization mode; `vec_mat_mul` NAMES the mode and the source returns that tap at the conversion shape. The reference source bills static area and leakage only — the conduction of the reference current through the compare branches is the readout's, inside `E_fixed` and `E_code`.

**PH0 is CONFIGURED, not derived.** `i_ph0_comp__uA` is a required macro config field that the macro hands the readout at construction. It is a calibration product, measured as the array's all-off row leakage: every physical column of a row carries the off-cell floor, weighted by its plane's place value, and one row's worth of that is what the readout subtracts once per conversion. Seated at the measured leakage, a zero-MAC access lands on code 0 and the redundant plane's floor is inside what the seat removes. Nothing outside the cell reads the cell's `I_T2` table, so the floor entries carry no cross-module constraint. PH0 compensation is static by design: no replica, dummy, or tracking circuit exists in the paper, so a small activity-dependent over-subtraction remains as a model PREDICTION, not an error.

**The design ships ONE quantization mode.** `config.modes` declares it — the canonical MAC-unit window the readout covers, the input code range it discriminates, and the rescale factor of a code at the readout's `bits` — and `quantization_mode` indexes that list. The mode set and the reference bank's mode set are the same size, one tap per mode; the readout has ONE physical current step, so this design point's single tap carries it. A request below `adc_max_bits` is realized inside the readout, which converts at full resolution and then drops the code's unresolved low bits — the same code the truncated compare-phase sequence resolves. The macro has no lossless oracle: `adc_bits = None` raises, and the exact-integer twin comes from `to_ideal()`.

**Timing.** `t_phase__ns` carries PH0 (the settling / leakage-compensation phase) followed by one compare phase per bit, MSB-first, at their physical durations. Each compare phase carries its OWN latch delay in `t_intrinsic__ns` — `t_1 .. t_B` MSB-first, the delay from a phase's start to that phase's comparator output latching. ONE BIT resolves per compare phase, so a conversion at `b` bits runs PH0 and the first `b` compare phases and closes when that last EXECUTED comparator output latches, `t_b` into its own phase. The access window is DERIVED per resolution:

$$T_{\mathrm{AC}}(b) = t_{\mathrm{PH0}} + \sum_{i=1}^{b-1} t_{\mathrm{phase},i} + t_{b}$$

At `b = 1` that is PH0 plus the FIRST compare phase's latch delay; at `b = adc_max_bits` the readout runs its whole phase set, and that maximum-resolution window is the NOMINAL access window the macro publishes as `t_ac__ns`. Within one resolution the window is code-independent: there is no data-dependent early termination and no pipelining. `latency__ns` reports that same executed window: the phase axis is the readout's own, so it is the readout that counts it.

**Energy.** `E = E_fixed(b) + E_code(b)`, both following the EXECUTED phases. `E_fixed` is the dominant, code-independent per-conversion baseline — the standby reference branches, the bias network, the comparator, and the leakage-compensation branch — held for as long as the conversion occupies the readout, so it is apportioned by the executed-window ratio `T_AC(b) / T_AC(B)` with `B = adc_max_bits`. `E_code` sums over the executed compare phases; phase `p` weighs the SAR residue `I_COMP` against that phase's place value on the ONE injected reference, `I_REF,p = 2**(B - p) * I_ref`, and the comparator input mirror draws a fraction `k` (`mirror_scale`) of the compared branch current from `v_rail__V` for that phase:

$$E_{\mathrm{code}}(b) = k \cdot V_{\mathrm{rail}} \sum_{p=1}^{b} t_{\mathrm{phase},p} \cdot \min(I_{\mathrm{COMP},p}, I_{\mathrm{REF},p})$$

The residue drops by the reference only where the bit resolves 1, and the recursion truncates with the executed phases — the unresolved low bits are exactly the ones no phase compares. The latched reference-subtraction branches are NOT billed on top: REFS sources exactly the current the mirror-input branch stops drawing, so the swap is rail-energy-neutral. The last executed compare phase's energy is billed over its full nominal duration even though the access window closes at the latch — the compared branch conducts until the phase-boundary reset.

## Dataflow

`vec_mat_mul` runs the whole readout as one broadcast tensor pipeline over the outputs, with no Python output-loop:

1. **Tile the inputs plane-major** into the physical BL columns (`[..., row_num] -> [..., plane, row_num] -> [..., phys_col]`) and hand the per-column input bits to the BL converter bank, which drives each column at that bit's level. The redundant planes are forced to input-0.
2. **Stack the output-serial one-hot word lines** on the leading: output `o` raises word line `o` and every other line holds its deselected code, and the WL converter bank drives the pattern; the same per-column inputs broadcast to every output. The word-line drive is expanded (stride-0) across the physical columns into the full cell grid the array reads, which is also the macro's declaration that one scanned row carries no per-column structure of its own.
3. **Snapshot both boundary clamps and solve once**. The macro owns the event structure, so the macro snapshots: each clamp is sampled at the full event shape `(..., out, *inst_shape, phys_col)`, one draw per output access, with the output axis ahead of each bank's own instance block. The solve is one broadcast call; the leading becomes `(..., out, *inst_shape)`. It returns both boundaries' per-column port current and clamp voltage, and the raw per-output summed T2 current. The macro then drives its own BL and SL clamps at that port state, on the layout the solve returned — before the output axis moves — so each clamp's per-op lump seats on its own column instance axis.
4. **Conduction branches** (macro-billed) — see the energy model below.
5. **RS-CSA quantize** the raw TBL current into the unsigned code, against the reference current the mode's `Iref` tap supplies.
6. **Peripheral energy**: bill the flat per-output-access mux-driver and timing-control lumps; see Energy below.

## Transfer

Only the selected row conducts, and inside it every physical column sits at one of the two operating points: the input-high weight columns at the drive point, the input-low columns and the all-HRS redundant plane at the floor $I_{0}$. The RAW row current is the place-value-weighted sum of those entries, and the readout subtracts the seated PH0 once. With $M$ the logical MAC of one output, $A$ the number of raised inputs, $R$ the sum of `weight_radix`, $P$ the sum of the full radix list, and $N$ the row count:

$$I_{\mathrm{COMP}} = I_{\mathrm{LRS}} M + I_{\mathrm{HRS}} (R A - M) + I_{0} (P N - R A) - I_{\mathrm{PH0}}$$

The second term is the complementary leakage: the raised inputs present $R A$ place-value units in all, of which $M$ conduct at LRS and the rest at HRS. The third is the floor every remaining place-value unit of the row carries, the redundant plane included. Seating $I_{\mathrm{PH0}}$ at the all-off row leakage $I_{0} P N$ reduces the compensated current to each cell's EXCESS over the floor on the input-high columns alone, and a zero-MAC access to code 0. The quantizer is uniform with the injected reference as its step, so the code is `floor(I_COMP / I_ref)` clamped to the ladder. This closed form, built from the config's own tables and the configured seat, is the golden-transfer gate's oracle: it reaches the code analytically, while the macro reaches it through the solve, the per-plane radix sum, the PH0 subtraction, and the readout's own compare phases.

## Energy model

Two atoms. A conduction branch is one rail-to-GND path, `E = V * I(uA) * t(ns) = fJ`, billed across the rail the charge LEAVES — a node's own level never appears in a branch bill. A capacitance is billed on the SUPPLY-DRAW law, `E = V_rail * C * |dv|` — the charge a node's excursion pulls out of the rail of the driver that moves it, counted on the charging leg of the round trip only. The array bills every capacitance inside it and nothing else, each converter bank bills its own drive events, and the macro bills the two conduction branches.

**Two conduction channels, both over the whole access window.** The design holds its DC biases for the entire access — the paper's phase diagram shows no sample-and-hold in the CIM path — so both branches conduct for the EXECUTED window `T_AC(b)` of the call's `adc_bits`, which shortens with the compare phases the readout skips:

- `bl_cond` = `v_dd_bl__V * I_BL * T_AC(b)`, summed over columns and outputs — the BL driver's supply sourcing every column's divider through the clamp. A column held at the IN=0 level carries no port current and self-zeroes.
- `dl_cond` = `v_dd_core__V * I_TBL_raw * T_AC(b)`, summed over outputs — the RAW row current entering the RS-CSA, leakage floor included, before the PH0 subtraction, riding the core rail.

**Two converter banks, each billing its own drive events.** Both drives are 1-bit, and each bank carries a per-code energy table, so a code pays what driving that level costs and a deselected or IN=0 line pays its own entry rather than nothing:

- `wl_dac` holds one seat per word line and fires every seat once per output access — one selected code and the rest deselected, on every die and for every input vector.
- `bl_dac` holds one seat per physical column and fires every seat once per input VECTOR, because the level is held across the whole row scan.

**The capacitance model: a held bit line, a scanned word line.** The access timing is return-to-zero on both axes. The scanned word line runs zero → drive → zero on every access. The held BL pattern runs zero → drive → held across the whole scan → zero, once per input vector, which is the semantics of one `vec_mat_mul`. The array's billing follows exactly that, as the `bl_in_wl_scan` organization of the kernel array:

- **Rest levels are IDEAL, never solved.** Between accesses the BL rests at the per-column level its converter holds, the SL at its grounded reference, and the WL at 0 V. An IR drop is what a solve produces, not what the array parks at.
- **The ledger is per NODE.** The array states one capacitance per cell node — the cell's own junction plus that node's share of the line it hangs on — and every stretch of line belongs to the node it hangs on, so the four per-node totals are the whole account. There is no separate wire ledger to reconcile against.
- **Per access** the array bills `V_rail * C * |v_dcop - v_rest|` over each of the four nodes of every cell: BL, the internal node X, SL, and the gate. X's rest level is the BL boundary itself: with T1 off it is orders of magnitude less conductive than the divider's BL-side leg, so X is pinned to `V_BL`; turning T1 on for the selected row pulls X down by `vx_ratio_on * (V_BL - V_SL)`, and that displacement is the dip the BL rail recharges. Input-low columns sit at 0 V throughout and contribute nothing.
- **Per hold** the whole rest state has to be established from ground once. One hold covers exactly ONE full row scan here — a call drives one BL pattern and selects each of the `row_num` array rows once, because the output axis the macro serializes IS the array's row axis — so the array amortizes that establishment over the scan, billing each access one `row_num`-th of it. The word line rests at ground in either organization, so it carries no precharge term.

**Two driver rails.** `v_dd_wl__V` is behind the WL node totals; `v_dd_bl__V` is behind the conduction-path node totals (BL, X, SL) and carries the macro's BL input conduction branch. They are separate variables even when numerically equal, and both are separate from `v_dd_core__V`, which is the READOUT's rail and carries the DL conduction branch: what a charge is drawn from is the supply of the driver that moves it.

The array's node ledger is the sole account of the capacitance inside the array: no macro channel re-bills a BL column, so one excursion is one bill. The TBL / DL node capacitance is unmodeled: the selected row's TBL is co-driven with its word line and then held at the readout clamp's DC level for the access, and the array declares no capacitance on that node.

**Flat peripheral seats.** Mux & Driver and Timing & Mode Ctrl are `UnmodeledBlock` static-PPA seats. Their measured block powers are flat across the published operating points, so they are billed as pure static leakage; the macro's `mux_driver` / `timing_ctrl` dynamic channels exist for a design point that needs a per-op share and are zero at the paper's. Those channels are per-op lumps, not timed branches, so they do not follow the access window.

**Latency.** `latency__ns` is `col_num * T_AC(b)`: the ONE time-shared readout converts the logical outputs one after another, so the output axis is the macro's own time axis, and the activation is 1-bit, so no input-bit axis multiplies it. The macro does not sum its children — `T_AC(b)` is a whole-circuit window that already spans the array solve the compare phases run over — so it multiplies the readout instead. A caller's own batch or time position is an independent unit operation and does not multiply it.

## Paper circuit to modeling map

| Paper block | Our composition | Billing owner |
|---|---|---|
| WH-2T1R cell array (T1+RRAM divider + T2 source) | `array` — `Ye2023Jssc2t1rArray` (the kernel `XbarArray1t1r` plus the place-value sum) over `Ye2023Jssc2t1rCell`, transposed and digit-folded, fused step1 solve + step2 `i_t2__uA` sum | `array` bills every per-node capacitance under the held-BL scan law (per-access displacement + the amortized hold); the macro bills `bl_cond` and `dl_cond` |
| WL driver | `wl_dac` — a `Vdac` bank, one seat per word line | `wl_dac` self-bills its per-code drive, once per line per output access |
| BL input driver | `bl_dac` — a `Vdac` bank, one seat per physical column | `bl_dac` self-bills its per-code drive, once per column per input vector |
| BL input clamp | `bl_driver` — a per-column `VoltageDriver` (ideal `r_out = 0`; the wire IR drop is the array's) | static seat plus its own per-op interface-energy row, billed when the macro delivers it at the port state the array's solve returns; the conduction is the macro's, on `bl_cond` |
| SL drive | `sl_driver` — a per-column grounded ideal `VoltageDriver` | static seat plus its own per-op interface-energy row, billed at the same delivery seam; the grounded rail carries no conduction branch |
| RS-CSA (reference-subtracting current SAR) | `rscsa` — `RsCsaIadc` (uniform quantizer + the seated static PH0), single time-shared instance | `rscsa` self-bills `E_fixed + E_code` per conversion |
| RS-CSA reference generation | `rscsa_reference` — a single-tap `Iref`, one tap per quantization mode | static seat only (the reference branches are inside the readout's `E_fixed` / `E_code`) |
| Mux & Driver | `mux_driver` — an `UnmodeledBlock` seat | the seat's static leakage; the macro's `mux_driver` channel carries any per-op share |
| Timing & Mode Ctrl | `timing_ctrl` — an `UnmodeledBlock` seat | the seat's static leakage; the macro's `timing_ctrl` channel carries any per-op share |

## Parameter provenance and the free set

Every value of `params.toml` and `anchors.toml` that names a physical quantity carries a provenance tag, so a reader can separate what the paper states from what the model assumes or solves. The tag legend and its coverage rule are the campaign convention's, in [validation campaigns](../../../../validation/campaigns.md#provenance-tags).

The SANCTIONED free set is exactly three entries — four numbers — and every other field is pinned by its tag:

- **`mirror_scale`** (the RS-CSA `k`) and **`e_fixed_per_op__fJ`** — solved jointly against ONE constraint pair: per-conversion energy flat within tolerance of the measured anchor at both sparsity points, and per-code energy spread inside the measured window.
- **the two `[transcribed]` seat powers** — the Mux & Driver and Timing & Mode Ctrl block powers, adopted verbatim because they are flat across the published points.

The array's four per-node capacitances are pinned priors, and the WL node total carries the C_WL knob the campaign checks against its declared band, at the array-pin power of the dense sparsity point. Bounds, solved values, residuals, and the knob's saturation status live in `validations/ye2023jssc/results.md`; the numbers themselves live in `params.toml`.

## Scope

- **Macro-only, UNSIGNED.** The macro owns unsigned logical-to-plane encoding. Signed-weight representation above this unsigned logical domain remains a unit concern.
- **Single precision, single quantization mode.** No memory (read/write) mode; the write path costs no energy and no time. The quantization mode axis has one entry.
- **RSM mapping unused.** The SUBA4 slice is physically present and contributes its radix-weighted leakage, but the redundant-slice mapping algorithm itself is not modeled.
- **No mismatch, noise, or jitter of any kind.** No scheme module declares a sigma or a stochastic source; the sanctioned `all_off` policy is the only intended policy, and every intrinsic circuit non-linearity (the divider, the per-state `I_T2`, the floor-bucketize, the PH0 subtraction) is still computed. `all_off` is the noiseless, mismatch-free reference, not an idealized or zeroed model.
- **No network-accuracy target.** Network-level accuracy is out of macro scope.

## Config, docs, and validation

The citable paper design point lives under `validations/ye2023jssc`: `params.toml` (biases, the nested WH-2T1R array with its cell tables, per-node capacitances, rail links and solver, the two converter banks, the RS-CSA with its reference source and its PH0 compensation seat, and the flat peripheral seats — every field provenance-tagged), `policy.toml` (all-off), `anchors.toml` (hard-gate targets separated from ungated reference targets, plus the declared workload conventions and the leakage integration window the harness reports over), and `validate.py`. The campaign gates on five hard checks — the golden transfer with its asymmetric-value digit-order regression, the per-plane I_TBL table and its HRS bound, the RS-CSA per-conversion energy and code spread, zero-input decoding to code 0, and the derived access window — and reports the per-block power breakdown under a dual-caliber attribution WITHOUT gating on it, because the two published array points are mutually inconsistent under any single caliber. The outcome of the current run is recorded in `validations/ye2023jssc/results.md`. This document and `params.toml` are the scheme's spec. See the [validation convention](../../../../validation/campaigns.md).

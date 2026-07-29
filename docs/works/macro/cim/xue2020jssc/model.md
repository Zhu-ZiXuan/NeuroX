# Xue 2020 JSSC CIM macro

`Xue2020JsscCimMacro` models one 256x512 sub-array of the 1-Mb embedded ReRAM CIM macro of Xue et al. (IEEE JSSC 2020). The constructor binds its logical `input_num` and `output_num` arguments to internal `row_num` and `col_num`; `program` consequently accepts a logical signed weight matrix `[row_num, col_num]`. Internally, each weight is encoded as a sign plus `w_digit_num` radix-`w_digit_radix` magnitude digits and carried by `w_digit_num * 2` physical cells — a P (PWG) and an N (NWG) cell per digit. An `input_bit_num`-bit (K-bit) activation drives K serial single-bit WL sub-phases, LSB first.

The scheme lives in package `neurox.works.macro.cim.xue2020jssc`: `array.py` holds the scheme-local `SerialColumnXbarArray` (the column-MUX-serialized 1T1R array core), `dswct.py` / `sinwp_sc.py` / `pn_isub.py` hold the three readout reporter-leaf modules, `tmcsa.py` holds the `Tmcsa` phase-resolved conversion-billing reporter leaf, and `macro.py` holds `Xue2020JsscCimMacro` composing them into the readout chain driven by `vec_mat_mul`.

## Generalization

The paper design point (128 signed 3-bit weights = sign + 2 radix-2 magnitude digits, 2-bit sample-and-hold inputs) is **one config point**, not a hardcoded shape. The macro supports arbitrary `w_digit_num >= 1`, `w_digit_radix >= 2`, and `input_bit_num >= 1`; the vectorized readout degenerates cleanly at size-1 digit or bit axes:

- `w_digit_num = 1` is a single P/N digit with no cross-digit combine (the DSWCT digit-sum collapses to identity).
- `input_bit_num = 1` runs the live bit alone, using the MSB combine ratio directly with the sample-and-hold leg off (no sampled leg).
- `w_digit_radix > 2` requires a radix-level linear-conductance table in the cell config (the config author's responsibility).

The DSWCT and SINWP-SC ratios are derived DOWNWARD from two MSB anchors (`dswct_ratio_msb`, `sc_ratio_msb`), so any (D, K) is expressible from the two anchor scalars. The anchor values, and every other design-point number, live in `validations/xue2020jssc/params.toml`.

## Idealizations

- **Wire resistance is small but positive — a config choice, never a code hardcode.** A physical parameter may never be baked into the code as an assumption, so the BL/SL wire resistance is a required field of the nested array config (an `[assumed]` physical estimate; the paper reports none). The solver needs `R > 0` (it works with conductance `g = 1/R`), so a small positive segment/first-segment resistance is the sanctioned way to approximate a near-ideal wire. The solver then computes the position-dependent IR drop, so each cell's `V_BL` droops below the clamp reference by its wire drop — the array is **not** zero-wire.
- **The CABLC clamp is an ideal Thevenin source (`r_out__MOhm = 0`).** The BL is driven toward the clamp reference — the sole tap of the macro's dedicated single-tap `Vref` (`cablc_vref_config`, the paper's graph-read V_BLC), read per solve and injected as the `bl_driver`'s reference — while the wire IR drop stays the array's, not the clamp's. The SL is a grounded ideal driver (`V_SL = 0`): a direct ground tie whose reference is a plain 0 V tensor, no reference source.
- **Every physical column is an independent BL/SL ladder, solved on its own.** The macro computes an explicit slot map — a bijection `(slot, io, P/N, w_digit) -> physical column index` — placing every physical column into its column-MUX (slot, driver-lane) seat; `SerialColumnXbarArray` seats its cells by this map and settles all of them in one broadcast DC solve, the slot axis riding the solve's leading batch. An off column (outside its active slot) is DEFINED as grounded (BL clamp `v_ref = 0, r_out = 0`, SL at 0, WL held by the plane), so its voltages, current, and cap energy are identically zero and it is never materialized: summing the solved `(slot, driver-lane)` entries IS the full physical total.

No mismatch or noise is modeled anywhere; the sanctioned `all_off` policy (every child sub-policy at its lossless baseline) is the only intended policy. The CMD interface capacitance is not modeled: its precharge is folded into `e_control_per_op__fJ` under the `control` channel. The TMCSA is a **single-ended** current SAR — it returns an unsigned magnitude code, and the sign is assembled by the macro; the single-ended and differential current ADCs are parallel classes and the differential variant is not used here.

## Paper circuit to modeling map

Each paper block, its NeuroX composition, who bills its per-op energy, and the conduction window that energy integrates over. The energy atom is one rail-to-GND branch `E = V_DD * I * t`; branches are billed independently (p.204: the read-path current is the sum of the column currents and the mirrored currents, so each stage is an additive `V_DD` draw) and never double-counted.

**Branch-tensor law** (binding on every billing site): energy may be billed ONLY from explicitly materialized physical-branch current tensors. If a conducting branch's current is not needed by the value path, it is materialized anyway, and value and energy consume the SAME tensor. Interface tensors may be billed directly only when the interface IS the series branch (the macro `cablc` whole input branch). At a current mirror the interface tensor is gate-voltage-mediated information, not a branch current: value paths are scaling-placement-invariant (a sum of `s_k * i_k`), energy paths are not (per-leg windows differ).

| Paper block | Our composition | Billing owner | Window |
|---|---|---|---|
| Row array (1T1R cells + BL/SL wire) | `array` — `SerialColumnXbarArray`, composing the kernel `XbarCell1t1r` registry cell and `NestedParallelRailSolver`; the macro-computed slot map seats every physical column, one broadcast solve per WL sub-phase | `array` reporter leaf bills BL/SL/WL wire-cap + per-cell node-cap cycling only (no conduction); folds into the `cablc` slice | — (caps only) |
| CABLC (current-aware BL clamp) | `cablc` — the array's `bl_driver`, a `VoltageDriver` (`r_out__MOhm = 0`; device count `[io, P/N, digit]` via `inst_shape`); its clamp reference is the macro-owned `cablc_vref` (dedicated single-tap `Vref`), read per solve and injected | macro `cablc` channel bills the whole input branch `V_DD * I_DL` (the macro owns the per-bit conduction window) + `cablc_config` leakage seat | per-bit `window_array` |
| SL drive | `sl_driver` — the array's `sl_driver`, ideal `VoltageDriver` (`r_out__MOhm = 0`); its reference is a plain 0 V tensor (direct ground tie, no reference source) | leakage seat only (ideal, no conduction energy) | — (static) |
| WL driver | `wl_dac` — 1-bit ON/OFF `Vdac`; `latency_per_op__ns = 0` (folds into `t_cycle`) | leakage seat only | — (static) |
| DSWCT (down-scaling weighting mirrors) | `dswct` — `Dswct` reporter leaf; weights the per-digit BL legs by the place-value ratios `r_d` and sums over `w_digit` into `I_WDL` | `dswct` module self-bills conduction (`V_DD * abs(I_WDL)` per leg) + a `c_load` cap event per (slot x plane) + leakage seat | per-bit `window_array` |
| SINWP-SC (size-weighted sample-and-hold combine) | `sinwp_sc` — `SinwpSc` reporter leaf; materializes the per-leg currents `i_leg[k] = s_k * i[k]` (the mirror legs carry the `s_k`-scaled copies — p.206 Fig.7: held sinks X1/4, live sinks X1/2 of the DSWCT diode) and sums the SAME legs over the input-bit axis into `I_DL_PN` | `sinwp_sc` module self-bills the leg conduction `V_DD * sum(i_leg[k]) * window_sc[k]` (signed leg sum, not `abs`) + a `c_hold` cap event per (slot x bit) + leakage seat. The `i_dl_pn` branch (exchange switches + isub mirror input + SC sinks) is billed ONCE within the sc+pn pair — here | per-bit `window_sc` |
| PN-ISUB (P/N current netting) | `pn_isub` — `PnIsub` reporter leaf; `I_SUB = abs(I_P - I_N)` and the sign decision `I_N > I_P` | `pn_isub` module self-bills the three ISUB INTERNAL replica legs at 1:1 [assumed] (`V_DD * (I_P + I_N + I_SUB)` — NOT the input branch, NOT the ISUB output copy) + per-decision `e_per_op__fJ` + leakage seat | `t_other` |
| TMCSA (current SAR ADC) | `adc` — kernel `SarIadc` value converter, built with `enable_latency_record = False` AND `enable_energy_record = False` (energy-silent; keeps its real `step_latency__ns`) + `tmcsa` — the scheme `Tmcsa` billing reporter leaf | `tmcsa` module bills phase-resolved per step `s`: the materialized branches `i_ph2 = 3 * (i_sub + i_ref_path[s])` over `t_ph2[s]` and `i_ph3 = 2 * (i_sub + i_ref_path[s])` over `t_ph3[s]`, plus `e_fixed` per step; `i_ref_path[s]` recovered from the final code through a structural tap LUT. The `i_sub` output copy (isub -> CSA delivery) is the TMCSA input branch, billed HERE (the PH2/PH3 coefficients include it) | per-step `t_ph2` / `t_ph3` (inside each step latency; PH1/PH4 fold into `e_fixed`) |
| Reference (I_REF midpoint ladder) | `adc_current_reference` — `Iref`, a per-instance `[*inst, mode, tap]` bank sliced by `quantization_mode` to the `[*inst, tap]` max-bits ladder and passed whole into the ADC at every `adc_bits` | `reference_config` leakage seat only | — (static) |
| Control (address decode, CMD, timing) | `control` — `UnmodeledBlock` static seat | macro `control` channel (`e_control_per_op__fJ`, CMD precharge folded in) at the pure per-op caliber: the full adopted share is dynamic, the `control_config` leakage seat is zero | once per op |

DSWCT, SINWP-SC, PN-ISUB, and TMCSA are scheme-local `ModuleBase` reporter leaves — `Dswct`, `SinwpSc`, `PnIsub`, `Tmcsa` — each self-billing its own conduction and cap events on its own profiler row at the production site, under the conduction window the macro injects per call (drive-context pattern; the TMCSA phase windows are its own config constants); their config classes carry the required cap / per-op knob (`c_load__fF`, `c_hold__fF`, `e_per_op__fJ`, `e_fixed_per_op__fJ`) and static PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) as physical config seats, never a code default.

## Dataflow

`vec_mat_mul` runs the whole readout as one broadcast tensor pipeline over the K WL sub-phase planes, with no Python bit-loop:

1. **Bit-expand** the K-bit activation into K WL planes (plane `k = (x >> k) & 1`, LSB first); the WL DAC maps each plane to its ON/OFF voltage.
2. **Solve** all K WL planes in ONE broadcast solve through `SerialColumnXbarArray`: the linearized cells plus the BL/SL wire IR drop, driven by the `cablc` clamp toward the injected `cablc_vref` tap on the BL and the grounded `sl_driver` (0 V tensor) on the SL, yielding the per-lane BL port current `I_DL` (MAC-sum over rows) at the position-dependent clamp voltage, in the structured `[..., x_bits, serial, io, P/N, w_digit]` (slot, driver-lane) layout — the slot map places every physical column into this layout once, so no scatter back to a flat column axis occurs anywhere downstream.
3. **DSWCT**: weight the per-digit legs by the place-value ratios `r_d` and sum over `w_digit`, producing the per-bit contribution `I_WDL`.
4. **SINWP-SC**: weight the per-bit DSWCT outputs by the combine ratios `s_k` and sum over the input-bit axis into `I_DL_PN`.
5. **PN-ISUB**: `I_SUB = abs(I_P - I_N)`; `sign = I_N > I_P`.
6. **TMCSA**: `code = adc.convert(I_SUB, adc_refs_mode, bits)` against the per-instance reference ladder (`[*inst, mode, tap]` sliced by `quantization_mode` to the full `[*inst, tap]` max-bits ladder, broadcast against `I_SUB`); a lowered `adc_bits` truncates the converter's own binary search after its leading steps, and the `tmcsa` billing module bills the conversion from the same `(I_SUB, code, adc_refs_mode)` triple; `signed = (1 - 2*sign) * code`, transposed and flattened to the primitive trailing `[col_num]`.
7. **Control energy + latency**: bill the control-node dynamic energy and emit the sole latency event (`t_cycle * serial`); see Energy below.

DSWCT, SINWP-SC, and PN-ISUB run natively per-(slot, IO) throughout; only the final code assembly maps the `(slot, IO)` layout back to the logical column order.

## Transfer

The DSWCT mirror ratios `r_d = dswct_ratio_msb * w_digit_radix**(d - (D-1))` (LSB-first `d`) and the SINWP-SC leg ratios `s_k = sc_ratio_msb * 2**(k - (K-1))` (LSB-first `k`) are derived DOWNWARD from the two MSB anchors; their product `r_d * s_k` over (digit, bit) is the composite BL-to-`I_SUB` coefficient, which reproduces the paper's Fig.7 net transfer at the design point. The physical cell-column count is derived, never stored: `phys_col_num = col_num * w_digit_num * 2`, `io_num = col_num // mux_factor`. The macro computes the slot map `(slot, io, P/N, digit) -> phys_col` from this geometry and passes it to `SerialColumnXbarArray` at construction. `program` accepts the logical matrix, invokes the macro-owned true-form transcoder to the (P/N, digit) grouped layout, then routes it through the array's `program`, which seats every physical column at its slot-mapped position. A positive digit writes its magnitude into the PWG cell and leaves the NWG cell at HRS; a negative digit does the reverse. Codes are signed-magnitude `(1 - 2*sign) * magnitude`; the sign and offset live in the macro, not inside the ADC.

`config.modes` declares one quantization mode per reference ladder row: the canonical mid-zero MAC-unit window the mode covers, the inclusive magnitude code range the TMCSA discriminates (circuit knowledge — the sign never enters the converter), and the mode's rescale factor at `adc_max_bits`. `map_quantization_input_code` is consequently the magnitude map, and `to_ideal()` publishes `adc_max_bits + 1`, since a sign plus B magnitude bits spans a signed range a zero-point quantizer reaches only one bit wider. The twin stays deliberately unfaithful at the window bottom — a mid-zero window holds one level this encoding never emits, and a single zero where the readout has two — so comparing macro against twin measures that gap and is never an equality check. The macro itself has no lossless oracle: `adc_bits = None` raises.

## Energy basis and windows

The energy atom is the rail-to-GND branch `E = V_DD * I * t`, split into a dynamic term integrated over the short conduction windows and a static term integrated over the full operating period. The hard validation target is the paper's energy per access, derived in `validations/xue2020jssc/anchors.toml` from the reported macro power, the sub-array count, and the measurement clock.

- **Static energy uses the operating period as its time base.** The macro is the sole latency emitter: it logs one latency event `t_cycle * serial` (serial = `mux_factor` column-MUX accesses, `t_cycle__ns` = the measurement period), and the profiler's `leakage_energy = leakage_power * total_latency` is then the leakage over the whole period. The array and the WL DAC carry zeroed latency and the kernel ADC is built with `enable_latency_record = False`, so no child emits a latency event and nothing double-counts against `t_cycle`.
- **Dynamic energy uses the conduction windows only.** A K-bit input runs as K serial single-bit WL sub-phases, LSB first; the sampled bits (0 to K-2) settle over their own `t_sample__ns[k]` window and are held on the SINWP-SC caps, and the live bit (K-1) settles in the tail `t_other = t_settle__ns + sum(ADC step latencies)`.
  - `window_array[k]` — the input branch (array / CABLC / DSWCT) window: `t_sample__ns[k]` for a sampled bit, `t_other` for the live bit.
  - `window_sc[k]` — the SINWP-SC leg window: a held leg conducts from its sample sub-phase to the end, so it is the suffix sum `sum(t_sample__ns[k:]) + t_other`; the live leg reduces to `t_other`.
  - The SAR sensing durations (`adc_config.step_latency__ns`) enter `t_other` so the read chain conducts through sensing. The conversion's own energy is billed by the `tmcsa` module phase-resolved: within each step, PH2 conducts for `t_ph2[s]` and PH3 for `t_ph3[s]` (`tmcsa_config`, keeping the Fig.10(b) as-drawn PH2:PH3 ratio while the window widths are jointly calibrated with `e_fixed` against the paper's TMCSA slice — an open tension recorded in the `validations/xue2020jssc/README.md` contradiction table; per step `t_ph2[s] + t_ph3[s] <= step_latency[s]`, PH1/PH4 fill the rest and fold into the per-step `e_fixed`). The kernel ADC's B-form conduction knobs are inert in this scheme (`enable_energy_record = False`).

The whole input branch `V_DD * I_DL` is billed by the macro on the `cablc` channel (the macro owns the per-input-bit conduction window `t`); the `array` module row bills only its BL/SL/WL wire-cap + per-cell node-cap cycling, no conduction. Those caps are full-cycle `C * V^2` per activation: the BL/SL wire and cell node terms sum over the solved (slot, lane) entries, while the WL wire cap and the per-cell WL gate caps are billed once per PLANE — the row drive is held across all serial slots, so the WL term is independent of `mux_factor`. The validation slice map sums the array module row (caps) and the `cablc` channel (whole conduction) into one `cablc` slice, and pools the `tmcsa` billing-module row with the kernel `adc` static seat into the `tmcsa` slice. `t_cycle__ns` must contain the whole conduction span `sum(t_sample) + t_other`; the read path idles for the remainder of the period.

### Paired-slice comparison caliber

The paper's Fig.18 pie splits ONE series input branch at node V_CMD (the drain of the DSWCT current-mirror input, p.207 Fig.9(a)) between the DSWCT and CABLC slices, and one series sink branch between SINWP-SC (its sink transistors) and PN-ISUB (switches + comparator + isub). The internal node voltages are not published, so only the pair sums are well-defined comparison targets: the validation breakdown compares `cablc+dswct` and `sinwp_sc+pn_isub` against the summed shares of their members, with `control` / `reference` / `tmcsa` as singles and the four member rows reported informationally (no per-member target). The shares themselves are `validations/xue2020jssc/anchors.toml`.

### Per-access normalization

The paper's power measurement counts access cycles: one access is one column-MUX slot conversion set (the `io_num` CIM-IOs conducting in parallel), and a full `vec_mat_mul` over all `col_num` outputs is `mux_factor` serial accesses. Dividing the reported macro power by the sub-array count and the measurement clock therefore yields the per-access energy that is the hard validation target.

## What the paper leaves underdetermined

The paper reports no internal node voltage, no per-block boundary, and no absolute device parameters, so several quantities are declared or seated, never fitted to pass:

- the CMD node voltage `V_CMD` (and hence the CABLC/DSWCT node-voltage split);
- the per-block silicon boundaries;
- the DSWCT cascode/bias legs;
- the PN-ISUB bias and common-mode legs;
- the WL on-voltage and the sample/settle sub-phase durations;
- the BL/SL wire resistance (an `[assumed]` small positive config estimate);
- the absolute conductance map `g_map` and the device R-ratio (the paper reports neither an absolute `g_LRS`/`g_HRS` nor the fabricated device's ratio; Fig.17 on p.210 only sweeps read yield over R-ratio 10, 15, 20, 30, 40, 50) — both assumed values, the ratio picked mid-sweep.

Every such item is a config seat or a declared constant.

## Config and validation

The citable paper design point lives under `validations/xue2020jssc`: `params.toml` (the paper geometry, anchors, window and seat knobs, the nested array with the Linear cell and wire R/C and the solver), `policy.toml` (all-off), `anchors.toml` (the hard energy-per-access target, the Fig.18 slice shares, and the declared dyn/static and data conventions), and `validate.py` (run it as `make validate_xue2020jssc`; the three TOML artifacts are fixed files beside the script and only the run knobs are CLI-settable). The campaign gates on total energy per access only; the per-block breakdown is reported but informational (not gated). The outcome of the current run is recorded in `validations/xue2020jssc/results.md`. See the [validation convention](../../../../validation/campaigns.md).

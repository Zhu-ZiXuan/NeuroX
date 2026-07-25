# Xue 2020 JSSC CIM macro

`Xue2020JsscCimMacro` models one 256x512 sub-array of the 1-Mb embedded ReRAM CIM macro of Xue et al. (IEEE JSSC 2020). The constructor binds its logical `input_num` and `output_num` arguments to internal `row_num` and `col_num`; `program` consequently accepts a logical signed weight matrix `[row_num, col_num]`. Internally, each weight is encoded as a sign plus `w_digit_num` radix-`w_digit_radix` magnitude digits and carried by `w_digit_num * 2` physical cells — a P (PWG) and an N (NWG) cell per digit. An `input_bit_num`-bit (K-bit) activation drives K serial single-bit WL sub-phases, LSB first. The scheme is a single self-contained macro: it composes the shared kernel `XbarArray1t1r` (a linearized 1T1R cell grid plus BL/SL wire parasitics and the DC solver) and runs the whole current-mode readout inline as vectorized tensor operations in `vec_mat_mul`.

## Generalization

The paper design point (128 signed 3-bit weights = sign + 2 radix-2 magnitude digits, 2-bit sample-and-hold inputs) is **one config point**, not a hardcoded shape. The macro supports arbitrary `w_digit_num >= 1`, `w_digit_radix >= 2`, and `input_bit_num >= 1`; the vectorized readout degenerates cleanly at size-1 digit or bit axes:

- `w_digit_num = 1` is a single P/N digit with no cross-digit combine (the DSWCT digit-sum collapses to identity).
- `input_bit_num = 1` runs the live bit alone, using the MSB combine ratio directly with the sample-and-hold leg off (no sampled leg).
- `w_digit_radix > 2` requires a radix-level linear-conductance table in the cell config (the config author's responsibility).

The DSWCT and SINWP-SC ratios are derived DOWNWARD from two MSB anchors (`dswct_ratio_msb`, `sc_ratio_msb`), so any (D, K) is expressible from the two anchor scalars.

## Idealizations

- **Wire resistance is small but positive — a config choice, never a code hardcode.** A physical parameter may never be baked into the code as an assumption, so the BL/SL wire resistance is a required field of the nested array config (an `[uncertain]` physical estimate; the paper reports none). The solver needs `R > 0` (it works with conductance `g = 1/R`), so a small positive segment/first-segment resistance is the sanctioned way to approximate a near-ideal wire. The solver then computes the position-dependent IR drop, so each cell's `V_BL` droops below the clamp reference by its wire drop — the array is **not** zero-wire.
- **The CABLC clamp is an ideal Thevenin source (`r_out__MOhm = 0`).** The BL is driven toward the clamp reference `v_bl_clamp__V` (paper V_BLC ~0.29 V) by the array's `bl_driver`; the wire IR drop is the array's, not the clamp's. The SL is a grounded ideal driver (`V_SL = 0`).
- **The array is column-separable**, so the grouped conductance layout folds into the array's flat `[phys_col, row]` layout and the column-MUX regroup is a pure reshape: each physical column is an independent BL/SL ladder solved on its own.

No mismatch or noise is modeled anywhere; the sanctioned `all_off` policy (every child sub-policy at its lossless baseline) is the only intended policy. The CMD interface capacitance is not modeled: its precharge is folded into `e_control_per_op__fJ` under the `control` channel. The TMCSA is a **single-ended** current SAR — it returns an unsigned magnitude code, and the sign is assembled by the macro; the single-ended and differential current ADCs are parallel classes and the differential variant is not used here.

## Paper circuit to modeling map

Each paper block, its NeuroX composition, who bills its per-op energy, and the conduction window that energy integrates over. The energy atom is one rail-to-GND branch `E = V_DD * I * t`; branches are billed independently (p.204: the read-path current is the sum of the column currents and the mirrored currents, so each stage is an additive `V_DD` draw) and never double-counted.

| Paper block | Our composition | Billing owner | Window |
|---|---|---|---|
| Row array (1T1R cells + BL/SL wire) | `array` — `XbarArray1t1r` (Linear cell grid + wire R/C + DC solver); grouped weights fold into the flat `[phys_col, row]` layout, solved once per WL sub-phase | `array` module row bills BL/SL/WL wire-cap + per-cell node-cap cycling only (no conduction); folds into the `cablc` slice | — (caps only) |
| CABLC (current-aware BL clamp) | `cablc` — the array's `bl_driver`, a `VoltageDriver` (`r_out__MOhm = 0`; device count `[io, P/N, digit]` via `inst_shape`) | macro `cablc` channel bills the whole input branch `V_DD * I_DL` (the macro owns the per-bit conduction window) + `cablc_config` leakage seat | per-bit `window_array` |
| SL drive | `sl_driver` — the array's `sl_driver`, ideal `VoltageDriver` (`r_out__MOhm = 0`, grounded) | leakage seat only (ideal, no conduction energy) | — (static) |
| WL driver | `wl_dac` — 1-bit ON/OFF `Vdac`; `latency_per_op__ns = 0` (folds into `t_cycle`) | leakage seat only | — (static) |
| DSWCT (down-scaling weighting mirrors) | macro tensor op — the place-value ratios `r_d` scale `I_DL` into `I_WDL` (linear current mirroring is KCL, not a block) | macro `dswct` channel (output legs `V_DD * abs(I_WDL)`); no static seat | per-bit `window_array` |
| SINWP-SC (size-weighted sample-and-hold combine) | macro tensor op — the per-bit ratios `s_k` weight the held/live legs (linear combining is KCL, not a block) | macro `sinwp_sc` channel (held/live legs `V_DD * I_DL_PN_k`); no static seat | per-bit `window_sc` |
| PN-ISUB (P/N current netting) | macro tensor op — `I_SUB = abs(I_P - I_N)` with a macro-assembled sign (a macro method, not a circuit class) | macro `pn_isub` channel (three branches `V_DD * (I_P + I_N + I_SUB)` + per-op comparator `e_pn_isub_per_op__fJ`) + `pn_isub_config` `UnmodeledBlock` seat (bias/icm + comparator static) | `t_other` |
| TMCSA (current SAR ADC) | `tmcsa` — `SarIadc` (B-form: `v_rail__V` + `t_conduct_per_step__ns`); built with `enable_latency_record = False` so it emits no latency event while keeping its real `step_latency__ns` | `tmcsa` module self-bills `e_fixed` per step plus per-step sensing conduction | per-step `t_conduct` (inside `t_other`) |
| Reference (I_REF midpoint ladder) | `adc_current_reference` — `Iref`, a per-instance `[*inst, mode, tap]` bank sliced by `adc_mode` to the `[*inst, tap]` ladder passed straight into the ADC | `reference_config` leakage seat only | — (static) |
| Control (address decode, CMD, timing) | `control` — `UnmodeledBlock` static seat | macro `control` channel (`e_control_per_op__fJ`, CMD precharge folded in) + `control_config` leakage | once per op |

DSWCT, SINWP-SC, and PN-ISUB carry no scheme-local class: the DSWCT mirror weighting and the SINWP-SC combine are linear current scaling and summation, which is Kirchhoff's current law realized directly as tensor operations; the PN-ISUB subtraction is likewise a tensor op, with its non-computational silicon (bias, common-mode, comparator) held by an `UnmodeledBlock` static seat and its conduction billed by the macro.

## Dataflow

`vec_mat_mul` runs the whole readout as one broadcast tensor pipeline over the K WL sub-phase planes, with no Python bit-loop:

1. **Bit-expand** the K-bit activation into K WL planes (plane `k = (x >> k) & 1`, LSB first); the WL DAC maps each plane to its ON/OFF voltage.
2. **Solve** each WL plane once through the array (`XbarArray1t1r`): the linearized cells plus the BL/SL wire IR drop, driven by the `cablc` clamp toward `v_bl_clamp__V` on the BL and the grounded `sl_driver` on the SL, yielding the per-column BL port current `I_DL` (MAC-sum over rows) at the position-dependent clamp voltage. The grouped conductance grid `[group_size(mux_factor), group_num(io), P/N, w_digit, row]` folds into the array's flat `[phys_col, row]`.
3. **DSWCT**: scale by the per-digit ratios `r_d` into `I_WDL`.
4. **SINWP-SC**: spatial sum over `w_digit` into the per-bit contribution `I_DL_PN_k`, then temporal weighted sum over bits with the ratios `s_k` into `I_DL_PN`.
5. **PN-ISUB**: `I_SUB = abs(I_P - I_N)`; `sign = I_N > I_P`.
6. **TMCSA**: `code = adc.convert(I_SUB, adc_refs_mode, bits)` against the per-instance reference ladder (`[*inst, mode, tap]` sliced by `adc_mode` to `[*inst, tap]`, broadcast against `I_SUB`); `signed = (1 - 2*sign) * code`, transposed and flattened to the primitive trailing `[col_num]`.
7. **Control energy + latency**: bill the control-node dynamic energy and emit the sole latency event (`t_cycle * serial`); see Energy below.

The column-MUX is a pure reshape (the mux slot rides the grouped axis), valid because the array is column-separable.

## Transfer

The DSWCT mirror ratios `r_d = dswct_ratio_msb * w_digit_radix**(d - (D-1))` (LSB-first `d`) and the SINWP-SC leg ratios `s_k = sc_ratio_msb * 2**(k - (K-1))` (LSB-first `k`) are derived DOWNWARD from the two MSB anchors, so the composite BL-to-`I_SUB` coefficient `r_d * s_k` over (digit, bit) is `(1/16, 1/8, 1/8, 1/4)` for the paper design (D=2, K=2). The physical cell-column count is derived, never stored: `phys_col_num = col_num * w_digit_num * 2`, `io_num = col_num // mux_factor`. `program` accepts the logical matrix and invokes the macro-owned true-form transcoder. A positive digit writes its magnitude into the PWG cell and leaves the NWG cell at HRS; a negative digit does the reverse. Codes are signed-magnitude `(1 - 2*sign) * magnitude`; the sign and offset live in the macro, not inside the ADC.

## Energy basis and windows

The energy atom is the rail-to-GND branch `E = V_DD * I * t`, split into a dynamic term integrated over the short conduction windows and a static term integrated over the full operating period. The hard validation target is **32.06 pJ per access** (the per-sub-array budget 5.13 mW / 8 sub-arrays / 20 MHz).

- **Static energy uses the operating period as its time base.** The macro is the sole latency emitter: it logs one latency event `t_cycle * serial` (serial = `mux_factor` column-MUX accesses, `t_cycle__ns` = the 50 ns = 1/20 MHz measurement period), and the profiler's `leakage_energy = leakage_power * total_latency` is then the leakage over the whole period. The array and the WL DAC carry zeroed latency and the TMCSA is built with `enable_latency_record = False`, so no child emits a latency event and nothing double-counts against `t_cycle`.
- **Dynamic energy uses the conduction windows only.** A K-bit input runs as K serial single-bit WL sub-phases, LSB first; the sampled bits (0 to K-2) settle over their own `t_sample__ns[k]` window and are held on the SINWP-SC caps, and the live bit (K-1) settles in the tail `t_other = t_settle__ns + sum(ADC step latencies)`.
  - `window_array[k]` — the input branch (array / CABLC / DSWCT) window: `t_sample__ns[k]` for a sampled bit, `t_other` for the live bit.
  - `window_sc[k]` — the SINWP-SC leg window: a held leg conducts from its sample sub-phase to the end, so it is the suffix sum `sum(t_sample__ns[k:]) + t_other`; the live leg reduces to `t_other`.
  - The TMCSA's SAR sensing durations (`adc_config.step_latency__ns`, paper 3.16/3.07/3.11 ns) enter `t_other` so the read chain conducts through sensing, and are mirrored by `t_conduct_per_step__ns` for the ADC's own `V_rail * (I_in + I_ref)` sensing-conduction energy.

The whole input branch `V_DD * I_DL` is billed by the macro on the `cablc` channel (the macro owns the per-input-bit conduction window `t`); the `array` module row bills only its BL/SL/WL wire-cap + per-cell node-cap cycling, no conduction. The validation slice map sums the array module row (caps) and the `cablc` channel (whole conduction) into one `cablc` slice. `t_cycle__ns` must contain the whole conduction span `sum(t_sample) + t_other`; the read path idles for the remainder of the period.

### Per-access normalization

The paper's 20 MHz power measurement counts access cycles: one access is one column-MUX slot conversion set (the `io_num` = 4 CIM-IOs conducting in parallel), and a full `vec_mat_mul` over all `col_num` outputs is `mux_factor` serial accesses. The per-sub-array budget 5.13 mW / 8 / 20 MHz is **32.06 pJ per access**, the hard validation target.

## What the paper leaves underdetermined

The paper reports no internal node voltage, no per-block boundary, and no absolute device parameters, so several quantities are declared or seated, never fitted to pass:

- the CMD node voltage `V_CMD` (and hence the CABLC/DSWCT node-voltage split);
- the per-block silicon boundaries;
- the DSWCT cascode/bias legs;
- the PN-ISUB bias and common-mode legs;
- the WL on-voltage and the sample/settle sub-phase durations;
- the BL/SL wire resistance (an `[uncertain]` small positive config estimate);
- the absolute conductance map `g_map` (the paper gives only the 10-50 R-ratio, not absolute `g_LRS`/`g_HRS`) — a declared value.

Every such item is a config seat or a declared constant.

## Config and validation

The citable paper design point lives under `validations/xue2020jssc`: `params.toml` (the paper geometry, anchors, window and seat knobs, the nested array with the Linear cell and wire R/C and the solver), `policy.toml` (all-off), `anchors.toml` (the 32.06 pJ/access hard target, the Fig.18 slice shares, and the declared dyn/static and data conventions), and `validate.py`. The campaign gates on total energy per access only; the per-block breakdown is reported but informational (not gated). See the [validation convention](../../../../validation/campaigns.md).

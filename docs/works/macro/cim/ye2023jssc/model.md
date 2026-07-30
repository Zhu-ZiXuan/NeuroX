# Ye 2023 JSSC WH-2T1R CIM macro

`Ye2023JsscCimMacro` models one compute-in-memory macro of the 28-nm RRAM CIM chip of Ye et al. (IEEE JSSC 2023), built on the Weighted Hybrid 2T1R (WH-2T1R) cell array and the Reference-Subtracting Current Sense Amplifier (RS-CSA). The constructor binds `input_num` and `output_num` to internal `row_num` and `col_num`; the interface is then `program(w[row_num, col_num])` followed by `vec_mat_mul(x[..., row_num], *, quantization_mode, adc_bits) -> codes[..., col_num]`. Logical unsigned weights are encoded internally into the configured binary weighted planes. The scheme composes the WH-2T1R lookup cell (`Ye2023Jssc2t1rCell`), the dedicated array (`Ye2023Jssc2t1rArray`) that solves over it, and the RS-CSA readout (`RsCsaIadc`) plus per-column BL/SL clamps and two flat peripheral seats.

The macro config carries no geometry: the logical dimensions arrive as constructor arguments.

## Geometry

The logical contract is realized on a TRANSPOSED, digit-folded physical grid — at the paper design point 64 rows × 128 columns, laid out as four 64 × 32 sub-arrays side by side:

- **Physical rows = logical OUTPUTS** (the TBL / ladder direction): one word line per output, driven one-hot.
- **Physical columns = `row_num * (len(weight_radix) + len(redundant_radix))`**, laid out plane-major: the weight planes first (SUBA1/2/3, T2 slice multipliers LSB-first, so `weight_radix[0]` carries digit 0), then the redundant plane last (SUBA4).

The place value of each plane is ANALOG: the radix-scaled T2 currents of the selected row's cells sum on one shared transpose bitline (TBL); there is no digital shift-add of bit-planes. Encoding a logical value lands each of its binary digits on the column group of the matching place value, so the digit order stays LSB-first end to end.

`weight_radix` and `redundant_radix` split the plane list into its two roles. The redundant plane carries no weight information: `program` writes it all-HRS and `vec_mat_mul` drives its BL columns at input-0, so it contributes only its radix-weighted share of the row leakage floor. The encode lookup table and `w_value_range` therefore span `weight_radix` alone, while the leakage path spans the full radix list.

## Design point and generalization

The paper design point (32 inputs, 64 outputs, three weight planes plus one redundant plane, 1-bit inputs, and a 4-bit RS-CSA) is one config point. The array supports any positive row count, any non-empty positive-integer `weight_radix` list, and any (possibly empty) `redundant_radix` list. The macro publishes the logical envelope `w_value_range = (0, sum(weight_radix))`; an arbitrary list may leave unrepresentable holes inside that envelope, and configuration validation deliberately does not impose a dense positional-radix pattern. The design is input-parallel and output-serial: `max_active_num == row_num`, while the outputs are produced sequentially by the single time-shared RS-CSA.

## Cell and the two-step decoupled solve

The WH-2T1R cell (`Ye2023Jssc2t1rCell`) is the shared linearized 1T1R cell plus ONE new physics field. T1 (an I/O device biased half-on as a resistor at the geometric mean of the two RRAM states) in series with the RRAM forms a divider that sets the internal node `V_X`; T2 (a core device in sub-threshold saturation) converts `V_X` into the compute current on the TBL. The macro splits this into two decoupled steps, both computed every call:

1. **step1 — the BL/SL divider solve.** The shared nested DC solver (`NestedParallelRailSolver`) settles the T1+RRAM divider driven by the per-column BL clamp (`v_bl_in1__V` for IN=1, 0 V for IN=0) against the grounded SL, yielding the per-column BL port current and `V_X`. The BL port current is what the macro bills as the input-branch conduction. The solved `V_X` itself feeds nothing downstream — neither the compute current nor any energy term reads it, so it is released with its chunk.
2. **step2 — the T2 compute current, by table lookup.** `I_T2` is a per-`(input bit, weight state)` lookup at unit slice scale: `(LRS, IN=1)` is the on-current `I_unit`, `(HRS, IN=1)` the unit-scale weight leakage, and `(any state, IN=0)` the off-cell floor `I_0` at `V_X = 0` — the per-cell contribution to the row leakage that PH0 compensates. The array scales each column's unit current by its plane's place value and sums over the active row into the raw TBL current `i_tbl`, floor INCLUDED. There is **no TBL solve, no VCCS, no device model**: T2 in sub-threshold saturation is a near-ideal current source, so the current delivered to the readout equals the sum by conservation regardless of TBL wire drop.

Sub-threshold off-state current scales with device width, so the radix multiplier applies to the leakage entries as well as to the on-current. The `(HRS, IN=1)` entry is set so that the LARGEST place value saturates the paper's measured per-plane HRS bound, which makes it a `[bound-derived]` value rather than a fit.

The two steps are FUSED per solver chunk (`Ye2023Jssc2t1rArray.solve`): `V_X` and other intermediates live only per chunk, so the array never persists the internal-node DCOP across a large leading batch. The output rows are solved HONESTLY (each output its own solve, batched over the leading dimension but not folded), because each row's wire geometry differs — the column-slot fold assumption does not apply to this serial-readout dimension.

## RS-CSA readout

The Reference-Subtracting CSA (`RsCsaIadc`) is a current-domain successive-approximation ADC that is functionally a UNIFORM quantizer over a decision ladder the owner supplies. The macro builds the ascending ladder `c * i_lsb__uA` for `c = 1 .. 2**bits - 1` and passes it in; the RS-CSA subtracts the static PH0 compensation current and floor-buckets the compensated current.

**PH0 is DERIVED, not configured.** The macro computes it from the array's own tables as `i_t2_table__uA[0][state] * row_num * sum(weight_radix + redundant_radix)` — every physical column of a row carries the off-cell floor, weighted by its plane's place value. A zero-MAC access therefore lands on code 0 by construction, and the redundant plane's floor cancels exactly. Config validation requires the `IN=0` floor row to be state-independent, since the derivation reads a single entry. PH0 compensation is static by design: no replica, dummy, or tracking circuit exists in the paper, so a small activity-dependent over-subtraction remains as a model PREDICTION, not an error.

**The design ships ONE quantization mode.** `config.modes` declares it — the canonical MAC-unit window the readout covers, the input code range it discriminates, and the rescale factor of a code at the readout's `bits` — and `quantization_mode` indexes that list. Every mode shares the one physical current step, so the `[mode, tap]` reference bank repeats the same max-bits ladder per row; the macro always hands the readout the full row, and a request below `adc_max_bits` is realized inside the readout, which converts at full resolution and then drops the code's unresolved low bits — the same code the truncated compare-phase sequence resolves. The macro has no lossless oracle: `adc_bits = None` raises, and the exact-integer twin comes from `to_ideal()`.

**Timing.** `t_phase__ns` carries PH0 (the settling / leakage-compensation phase) followed by one compare phase per bit, MSB-first, at their physical durations. ONE BIT resolves per compare phase, so a conversion at `b` bits runs PH0 and the first `b` compare phases and closes when that last EXECUTED comparator output latches, `t4_intrinsic__ns` into its phase. The access window is DERIVED per resolution:

$$T_{\mathrm{AC}}(b) = t_{\mathrm{PH0}} + \sum_{i=1}^{b-1} t_{\mathrm{phase},i} + t_{4}$$

At `b = 1` that is PH0 plus the latch delay alone; at `b = adc_max_bits` the readout runs its whole phase set, and that maximum-resolution window is the NOMINAL access window the macro publishes as `t_ac__ns`. Within one resolution the window is code-independent: there is no data-dependent early termination and no pipelining.

**Energy.** `E = E_fixed(b) + E_code(b)`, both following the EXECUTED phases. `E_fixed` is the dominant, code-independent per-conversion baseline — the standby reference branches, the bias network, the comparator, and the leakage-compensation branch — held for as long as the conversion occupies the readout, so it is apportioned by the executed-window ratio `T_AC(b) / T_AC(B)` with `B = adc_max_bits`. `E_code` sums over the executed compare phases; phase `i` weighs the SAR residue `I_COMP` against `I_REF = ref_radix[i] * i_lsb__uA`, and the comparator input mirror draws a fraction `k` (`mirror_scale`) of the compared branch current from `v_rail__V` for that phase:

$$E_{\mathrm{code}}(b) = k \cdot V_{\mathrm{rail}} \sum_{i=1}^{b} t_{\mathrm{phase},i} \cdot \min(I_{\mathrm{COMP},i}, I_{\mathrm{REF},i})$$

The residue drops by the reference only where the bit resolves 1, and the recursion truncates with the executed phases — the unresolved low bits are exactly the ones no phase compares. The latched reference-subtraction branches are NOT billed on top: REFS sources exactly the current the mirror-input branch stops drawing, so the swap is rail-energy-neutral. The last executed compare phase's energy is billed over its full nominal duration even though the access window closes at the latch — the compared branch conducts until the phase-boundary reset.

## Dataflow

`vec_mat_mul` runs the whole readout as one broadcast tensor pipeline over the outputs, with no Python output-loop:

1. **Tile the inputs plane-major** into the physical BL columns (`[..., row_num] -> [..., plane, row_num] -> [..., phys_col]`) and map each column to its BL voltage (`v_bl_in1__V` if IN=1, else 0). The redundant planes are forced to input-0.
2. **Stack the output-serial one-hot word lines** on the leading: output `o` activates array row `o` at `v_wl_sel__V`; the same per-column inputs broadcast to every output.
3. **One broadcast solve** through the WH-2T1R array; the leading becomes `(..., out, *inst_shape)`. Returns the per-column BL port current, the BL clamp voltage, and the raw per-output summed T2 current.
4. **Conduction + BL charge** (macro-billed) — see the energy model below.
5. **RS-CSA quantize** the raw TBL current against the uniform ladder into the unsigned code.
6. **Latency**: the sole event `T_AC(b) * serial`, where `serial` is the output-serial round count over the single time-shared readout and `b` is the call's `adc_bits`.

## Transfer

Every physical column carries the input-0 floor $I_{0}$, and the derived PH0 subtracts exactly one row's worth of it, so the current the quantizer resolves is the sum of each cell's EXCESS over that floor, taken over the input-high columns alone. With $M$ the logical MAC of one output, $A$ the number of raised inputs, and $R$ the sum of `weight_radix`:

$$I_{\mathrm{COMP}} = (I_{\mathrm{LRS}} - I_{0}) M + (I_{\mathrm{HRS}} - I_{0}) (R A - M)$$

The second term is the complementary leakage: the raised inputs present $R A$ place-value units in all, of which $M$ conduct at LRS and the rest at HRS. The redundant plane is driven input-0, so it cancels against PH0 and never enters. The code is `floor(I_COMP / i_lsb__uA)` clamped to the ladder, the quantizer being uniform. This closed form, built from the config's own tables, is the golden-transfer gate's oracle.

## Energy model

The energy atom is one rail-to-GND branch `E = V * I(uA) * t(ns) = fJ`. Branches are billed independently and never double-counted: the array bills CAPS only, and the macro bills every conduction branch plus the per-vector BL-column charge.

**Two conduction channels, both over the whole access window.** The design holds its DC biases for the entire access — the paper's phase diagram shows no sample-and-hold in the CIM path — so both branches conduct for the EXECUTED window `T_AC(b)` of the call's `adc_bits`, which shortens with the compare phases the readout skips:

- `bl_cond` = `v_bl_in1__V * I_BL * T_AC(b)`, summed over columns and outputs — the clamped input rail sourcing every column's divider.
- `dl_cond` = `v_dd_core__V * I_TBL_raw * T_AC(b)`, summed over outputs — the RAW row current entering the RS-CSA, leakage floor included, before the PH0 subtraction, riding the core rail.

**A three-part capacitance scan model.** The BL levels are held across the whole row scan, which is the semantics of one `vec_mat_mul`, while the word line toggles once per output access. The three parts split by who owns the toggling node:

- **array, per access**: the selected row's WL wire segments, `C_WL_wire * V_WL_sel^2`.
- **cell, per access**: the WL gate load of every cell on the selected row, plus that row's X-node dip-recharge — turning T1 on pulls X from `V_BL` down to `V_BL * (1 - vx_ratio_on)`, and the BL rail recharges it, `c_x__fF * V_BL^2 * vx_ratio_on`. The dip is closed-form in the PRESET per-state divider ratio, not in a solved node voltage. Input-low columns sit at `V_BL = 0` and contribute nothing.
- **macro, per vector** (`bl_cap`): each input-high column charges its BL wire and its `V_BL`-pinned cell nodes (the cell BL node plus the X node of every unselected row, whose T1 is off) exactly once per call.

The SL rail is grounded, so it carries no capacitive term and no conduction branch. The TBL / DL node capacitance is unmodeled: the readout clamp holds that node at a DC level for the whole access, so its switching share is negligible.

**Flat peripheral seats.** Mux & Driver and Timing & Mode Ctrl are `UnmodeledBlock` static-PPA seats. Their measured block powers are flat across the published operating points, so they are billed as pure static leakage; the macro's `mux_driver` / `timing_ctrl` dynamic channels exist for a design point that needs a per-op share and are zero at the paper's. Those channels are per-op lumps, not timed branches, so they do not follow the access window.

**Latency.** The macro is the SOLE emitter (the array and RS-CSA are built with `enable_latency_record = False`): one event `T_AC(b) * serial` per call, so the profiler's `leakage_energy = leakage_power * total_latency` covers the whole period without any child double-counting.

## Paper circuit to modeling map

| Paper block | Our composition | Billing owner |
|---|---|---|
| WH-2T1R cell array (T1+RRAM divider + T2 source) | `array` — `Ye2023Jssc2t1rArray` over `Ye2023Jssc2t1rCell`, transposed and digit-folded, chunk-fused step1 solve + step2 lookup/sum | `array` bills the per-access WL wire + WL gate + selected-cell X-dip caps; the macro bills `bl_cond`, `dl_cond`, and the per-vector `bl_cap` |
| BL input clamp | `bl_driver` — a per-column `VoltageDriver` (ideal `r_out = 0`; the wire IR drop is the array's) | static seat only (conduction billed by the macro on `bl_cond`) |
| SL drive | `sl_driver` — a grounded ideal `VoltageDriver` | static seat only |
| RS-CSA (reference-subtracting current SAR) | `rscsa` — `RsCsaIadc` (uniform quantizer + derived static PH0), single time-shared instance built with `enable_latency_record = False` | `rscsa` self-bills `E_fixed + E_code` per conversion |
| Mux & Driver | `mux_driver` — an `UnmodeledBlock` seat | the seat's static leakage; the macro's `mux_driver` channel carries any per-op share |
| Timing & Mode Ctrl | `timing_ctrl` — an `UnmodeledBlock` seat | the seat's static leakage; the macro's `timing_ctrl` channel carries any per-op share |

## Parameter provenance and the free set

Every value of `params.toml` and `anchors.toml` that names a physical quantity carries a provenance tag, so a reader can separate what the paper states from what the model assumes or solves. The tag legend and its coverage rule are the campaign convention's, in [validation campaigns](../../../../validation/campaigns.md#provenance-tags).

The SANCTIONED free set is exactly four entries — five numbers — and every other field is pinned by its tag:

- **`c_wl__fF`** — the per-cell word-line gate load, the sole capacitive knob, swept inside a declared C_WL band (fF per row) and solved against the array-pin power at the 50% sparsity point.
- **`mirror_scale`** (the RS-CSA `k`) and **`e_fixed_per_op__fJ`** — solved jointly against ONE constraint pair: per-conversion energy flat within tolerance of the measured anchor at both sparsity points, and per-code energy spread inside the measured window.
- **the two `[transcribed]` seat powers** — the Mux & Driver and Timing & Mode Ctrl block powers, adopted verbatim because they are flat across the published points.

Bounds, solved values, residuals, and the saturation status of the C_WL knob live in `validations/ye2023jssc/results.md`; the numbers themselves live in `params.toml`.

## Scope

- **Macro-only, UNSIGNED.** The macro owns unsigned logical-to-plane encoding. Signed-weight representation above this unsigned logical domain remains a unit concern.
- **Single precision, single quantization mode.** No memory (read/write) mode; the write path costs no energy and no time. The quantization mode axis has one entry.
- **RSM mapping unused.** The SUBA4 slice is physically present and contributes its radix-weighted leakage, but the redundant-slice mapping algorithm itself is not modeled.
- **No mismatch, noise, or jitter of any kind.** No scheme module declares a sigma or a stochastic source; the sanctioned `all_off` policy is the only intended policy, and every intrinsic circuit non-linearity (the divider, the per-state `I_T2`, the floor-bucketize, the PH0 subtraction) is still computed. `all_off` is the noiseless, mismatch-free reference, not an idealized or zeroed model.
- **No network-accuracy target.** Network-level accuracy is out of macro scope.

## Config, docs, and validation

The citable paper design point lives under `validations/ye2023jssc`: `params.toml` (biases, the nested WH-2T1R array with its cell tables, wire R/C and solver, the RS-CSA, and the flat seats — every field provenance-tagged), `policy.toml` (all-off), `anchors.toml` (hard-gate targets separated from ungated reference targets, plus the declared workload conventions), and `validate.py`. The campaign gates on five hard checks — the golden transfer with its asymmetric-value digit-order regression, the per-plane I_TBL table and its HRS bound, the RS-CSA per-conversion energy and code spread, zero-input decoding to code 0, and the derived access window — and reports the per-block power breakdown under a dual-caliber attribution WITHOUT gating on it, because the two published array points are mutually inconsistent under any single caliber. The outcome of the current run is recorded in `validations/ye2023jssc/results.md`. This document and `params.toml` are the scheme's spec. See the [validation convention](../../../../validation/campaigns.md).

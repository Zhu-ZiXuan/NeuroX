# Tile (IsubIadc1t1rCimMacro)

Scientific spec of the simplified ternary-weight crossbar tile: a registered CIM macro (`IsubIadc1t1rCimMacro`) composing the kernel [1T1R pure array](../../../../../reference/primitive/xbar/array/_1t1r/array.md) with the macro-owned boundary drivers and an inline kernel current-mode readout chain, running one independent conversion per WL plane. Provenance: the read-out chain of Xue et al., IEEE JSSC 2020, with input bit-serial sub-cycles, sample-and-hold, and weight digits removed.

## Value domain

- **Input** — binary codes, $x_i \in \{0, 1\}$ per row; one masked WL plane per conversion.
- **Weight** — a single ternary digit per logical column, $w \in \{-1, 0, +1\}$ (`w_digit_count = 1`, `w_digit_radix = 2`, `w_digit_range = (-1, 1)`).
- **Output** — a signed-magnitude code per logical column per WL plane: the sign bit from the subtractor and an `n_bits` unsigned magnitude from the ADC, decoded to a signed integer in $[-(2^{n} - 1), 2^{n} - 1]$ for $n$ = `adc_config.n_bits` (the width is a params value, not a code default). With a plane carrying at most $A$ = `active_row_num` live rows, the per-plane partial MAC $M_p = \sum_{i \in \text{plane } p} x_i w_i$ is recovered as $\mathrm{clip}(|M_p|, 0, 2^{n} - 1)$ with the subtractor's sign. Per-plane overflow clips to the top code — design intent (bounded per-conversion code domain), not a defect. A training-side floor quantizer of the same width spans one extra negative code (e.g. $-2^{n}$); that floor code is unreachable on this sign-magnitude hardware — a one-code floor-domain asymmetry.

## WL planes and the active-row window

One conversion drives at most `active_row_num` rows — the other rows are held at the WL off level (`row_num` must divide exactly by `active_row_num`). The planes arrive pre-masked from the consuming engine, whose sub-phase axis serializes a full-row read into `row_num / active_row_num` planes riding the anonymous broadcast leading; the tile runs the full analog chain once per plane and returns the code tensor with primitive trailing `[col_num]`, leading order preserved. There is no in-macro accumulation and no phase axis of the macro's own: reducing the codes over the engine's sub-phase axis is the consuming unit's digital-domain decision, and because each plane quantizes independently, $\sum_p Q(M_p) \ne Q(\sum_p M_p)$ in general — the per-plane quantization is the modeled physical semantics.

## Physical layout

Each logical column maps to two physical columns, so `phys_col_num = 2 * col_num`: logical column $c$ occupies physical column $2c$ (P) and $2c+1$ (N). Programming a weight of $+1$ sets the P cell LRS, $-1$ the N cell LRS, and $0$ leaves both HRS. Both polarities conduct simultaneously; the common HRS leakage cancels in the P-N difference.

## Array and boundary

The array physics is the kernel [1T1R pure array](../../../../../reference/primitive/xbar/array/_1t1r/array.md) (cells, wire parasitics, DC solver), programmed by state index and value-domain-agnostic. The boundary drivers are macro-owned peers of the array, injected into the array solve per call:

- **WL drive** — a 1-bit ON/OFF [VoltageDac](../../../../../reference/primitive/analog/voltage_dac/general.md); a zero code holds the row's word line at its off level (the unselected-row physical state within a WL plane).
- **BL boundary** — a current-aware clamp (the paper's CABLC slot) modeled as the Thevenin [VoltageDriver](../../../../../reference/primitive/analog/voltage_driver.md): $V_{BL} = V_{BLC} - I_{BL} R_{out}$ with finite $R_{out}$ for solver well-posedness.
- **SL boundary** — the same VoltageDriver as an ideal flat clamp.
- **Clamp reference** — one shared [VoltageReference](../../../../../reference/primitive/analog/voltage_reference.md) per tile sources both boundary levels, taps ordered $[V_{BLC}, V_{SL}]$; neither driver self-holds its reference. Snapshotted once per VMM.

The BL clamp is a column-MUX time-shared front-end lane, one per `mux_factor` logical columns per polarity, so its real instance count is `2 * n_lane` — this sizes only its PPA multiplicity and policy offset budget. The per-solve clamp still samples per physical column (one $V_{clamp}$ per BL), a conservative upper bound on offset diversity that is inert under the all-off policy.

## Signal chain

Per WL plane:

1. The binary WL plane arrives pre-masked to its active window (unselected rows at the WL off level); the WL DAC converts the codes to WL voltages and the kernel array settles the full fabricated array to DC with the BL held by the current-aware clamp and the SL by the ideal clamp, emitting the per-physical-column BL current.
2. The BL currents split into P/N polarity groups and regroup by front-end MUX lane.
3. The front-end [CurrentMirror](../../../../../reference/primitive/analog/current_mirror.md) applies the global $1/k$ down-scale (DSWCT slot).
4. The currents regroup by CIM-IO; the back-end [CurrentMirror](../../../../../reference/primitive/analog/current_mirror.md) applies the combiner normalization.
5. The [CurrentSubtractor](../../../../../reference/primitive/analog/current_subtractor.md) produces the sign bit and $I_{SUB} = \lvert I_{DL,P} - I_{DL,N} \rvert$ (PN-ISUB slot).
6. The [SarCurrentAdc](../../../../../reference/primitive/analog/current_adc/sar.md) quantizes $I_{SUB}$ to the `n_bits` magnitude against the mid-point threshold ladder of the per-call operating mode, sourced by the shared static [CurrentReference](../../../../../reference/primitive/analog/current_reference.md) (TMCSA + Reference slots).
7. The per-plane output code assembles per column: the magnitude negated where the sign bit is set.

## Transfer math and calibration

With mirror ratios $r_p$ and $r_n$ (per-stage values in `params/default.toml`) and unit subtractor gain, the ADC input per logical column per plane is

$$I_{SUB} = r_p \, r_n \, \lvert I_{BL,P} - I_{BL,N} \rvert .$$

The analog $I_{SUB}$ at a given per-plane $|M_p|$ is loading-dependent (clamp output resistance, wire IR drop, cross-column loading, and the plane's active-row position along the wires), so each ADC threshold centers the observed $I_{SUB}$ band of its code boundary over a representative calibration battery probed through the modeled chain (array through subtractor, up to the ADC input) on the all-off (lossless) policy:

$$I_{REF}[k] = \tfrac{1}{2}\bigl(\mathrm{hi}(k) + \mathrm{lo}(k+1)\bigr), \quad k = 0 \ldots 2^{n} - 2,$$

where $\mathrm{hi}(k)$ / $\mathrm{lo}(k)$ bound the band observed at per-plane $|M_p| = k$. The code then equals $\mathrm{clip}(|M_p|, 0, 2^{n} - 1)$ and the calibration rescale factor is exactly 1. Band separability is an electrical design requirement, not an assumption: the loading-dependent band spread scales with $I_{SUB}$ while the code spacing is one MAC LSB, so the array's series IR budget (strapped BL/SL read wires, the current-aware clamp's regulation residual sized for the full read-current range) must bound the spread below half an LSB at the top code — the probe reports the per-boundary band margins and all must be positive. The battery — a deterministic single-cell-LSB count grid realizing every magnitude exactly (full WL drive on a diluted column subset), count-capped random single-sign block patterns at full drive and moderate WL drive densities, and the fully-dense saturating extreme — spans the deployment loading envelope and converges under the array DC solve at the shipped operating point (probed by `neurox.tools.calibrate_adc.threshold_probe`; the run config beside the params records the battery, the run log records the band grid and margins). The probed thresholds live in `params/default.toml` (`adc_config.ref_levels__uA`, a 2-D `[mode][tap]` bank pinned equal to `reference_config.i_refs__uA` mode for mode — the reference block is the single source of truth; the per-call `adc_mode` selects the ladder row). **Operating modes**: the mode set is derived from a neutral per-layer range TOML (`neurox.tools.calibrate_adc.mode_derive` — signed/unsigned split by each layer's flag, then per-group range clustering; extracting the per-layer ranges from a trained checkpoint is a consumer-side step); every mode row that quantizes at the same MAC LSB carries the same physical ladder, and the per-mode rescale factors are fitted by `neurox.tools.calibrate_adc.rescale_fit`. **Calibration coupling**: `active_row_num` sets the per-conversion analog dot-product dynamic range, so the thresholds are valid only at the geometry and `active_row_num` they were probed at — re-probe whenever the array geometry, `active_row_num`, the conductance window, the wire budget, or the clamp residual changes.

## Sharing geometry

The boundary clamp and the readout circuits are column-MUX time-shared, not replicated per bit-line. Two knobs, both counting LOGICAL columns per polarity, derive every real device count:

| Knob | Derived count | Devices |
| --- | --- | --- |
| `mux_factor` | `n_lane = col_num // mux_factor` | BL clamp and front-end mirror: `2 * n_lane` each (P and N conduct simultaneously through separate devices) |
| `io_col_num` | `n_io = col_num // io_col_num` | back-end mirror: `2 * n_io`; subtractor and ADC: `n_io` each (one per IO, consuming both polarities) |

Divisibility is strict: `col_num` must divide exactly by `mux_factor` and by `io_col_num` (the lane / IO regroupings are exact reshapes; a non-divisor raises `ValueError`, never clamps). The static CurrentReference and the boundary-clamp VoltageReference are one per tile each. Because every count is geometry-derived, each block's per-instance PPA is a genuine per-circuit value and the totals scale with the real shared counts.

## Energy model

The array-internal terms — DC conduction at the boundary clamps, BL/SL wire-segment and WL-line capacitive cycling, per-cell device-capacitance switching — are specified and self-logged by the kernel [1T1R pure array](../../../../../reference/primitive/xbar/array/_1t1r/array.md). The macro, as the supply-rail and boundary-clamp owner, adds two array-side terms per solved WL plane (hence per engine sub-phase and per batch element):

- **Clamp drop** — the clamp-transistor dissipation $\sum_c (V_{DD} - V_{BL,c}) I_{BL,c} \, t_{pulse}$; together with the array's cell-side term this accounts the true rail draw $V_{DD} I_{BL}$. This term also owns the mirror input-leg conduction, so the readout chain never re-counts it.
- **CMD precharge** — the full-swing constant $\tfrac{1}{2} C_{CMD} V_{DD}^2$ per physical column (the BL to readout-front-end interface node, macro-owned).

Static clamp bias is folded into the BL clamp's leakage share, not a dynamic term. The readout-chain terms (mirror rails and bias floors, subtractor branches, ADC conversion) are specified per block and billed per plane.

## Serial-plane scaling of latency and energy

Per-op timing and energy seeds are per-op circuit properties; totals scale with the serial counts. Every serial count picks up the plane-count factor (`row_num / active_row_num` planes per full-row read, the consuming engine's sub-phase serialization): the array solves once per WL plane, each time-shared readout device serves its columns for every plane, and the ADC converts once per (IO, column, plane). Data-dependent rail energies follow the per-plane currents, and the per-op floors (mirror bias, subtractor replica branches, CMD precharge) bill once per plane — bias physically flows in every plane. The same physical devices serve every plane, so static PPA (area, leakage) does not scale with the plane count; leakage energy grows only through the longer total latency.

## Nonidealities

Each owned block carries its kernel nonideality policy: cell RRAM/NMOS sources through the array's cell cascade, boundary-driver offset/thermal (BL clamp, SL driver) and clamp-reference tolerance/noise, WL DAC drive noise, mirror ratio mismatch (independent P/N draws, static per fabricate), subtractor leg mismatch and sign-comparator offset, ADC comparator/coupling/mirror sigmas, and reference tolerance/noise — all gated by the tile policy (`policy/all_off.toml` is the lossless baseline). Sigmas are config fields; toggles are policy fields. Static fabrication mismatch is a property of the physical devices and is therefore shared across the planes a device serves; per-call noise draws independently per plane.

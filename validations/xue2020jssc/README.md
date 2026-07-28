# xue2020jssc Validation Config

Paper-design config, all-off policy, and calibration anchors for the
`Xue2020JsscCimMacro` SINWP 1T1R CIM sub-array — one 256×512 sub-array of the
1-Mb ReRAM CIM macro of Xue et al. (JSSC 2020). This directory holds the
citable design point the scheme is validated against; `validate.py` measures the
total energy per access against the 32.06 pJ/access target in `anchors.toml`, and
`tools/calibrate.py` re-derives the geometry-dependent seats (the ADC reference
ladder and the adopted peripheral seats) for `[calibrated]` write-back.

## Run

    make validate_xue2020jssc

The three TOML artifacts are FIXED files beside `validate.py`; the workload
`p_zero` comes from `anchors.toml`, never from the command line. Only run knobs
are CLI-settable, by invoking the script directly:

    TORCH_COMPILE_DISABLE=1 uv run python validations/xue2020jssc/validate.py --device cpu --n 32

`--n` (draws) / `--seed` / `--batch` / `--chunk-size` (large-N chunking) /
`--device` (`cpu`, `cuda[:idx]`, or `auto` for a free GPU) / `--sweep` (add the
`p_zero` sensitivity sweep + bracket). The harness logs its report; `results.md`
records the campaign run, whose header states its own draw count and seed.

## Files

- `params.toml` — the paper design: 256×512 sub-array, 32:1 column MUX (4
  CIM-IOs), 9-row (3×3-kernel) block, MSB ratio anchors `dswct_ratio_msb` 0.5 /
  `sc_ratio_msb` 0.5, K=2 input, 3-bit ADC, V_DD 1.0 V, V_BLC 0.29 V. Every value
  carries a provenance tag from the legend in
  [campaigns.md](../../docs/validation/campaigns.md): `[measured pN]` (paper
  page), `[derived]`, `[transcribed]` (the two Fig.18 peripheral seats),
  `[assumed]`, or `[calibrated]` (the cell chord, solver n_outer/n_inner, ADC
  ladder, and the energy-campaign caps + per-op constants).
- `policy.toml` — the all-off (lossless) policy; every nonideality toggle false.
- `anchors.toml` — the 5.13 mW hard target, Fig.18 shares, dyn/static
  conventions, data conventions, known-unknowns.
- `validate.py` — the profiler-driven gate driver (build → draw → energy per
  access → single hard-gate total + informational paired-slice breakdown +
  optional `p_zero` sweep).
- `tools/` — the calibration helpers: `calibrate.py` (ladder re-derivation +
  adopted-seat print) and `calibrate_cell.toml` + `params_detail.toml` (the
  Detail-cell chord source).
- `results.md` — the validation report `validate.py` renders; `calibration.md` —
  the calibration-run report `tools/calibrate.py` writes.

## Geometry

Derived, never stored: `phys_col_num = col_num · w_digit_num · 2 = 512` (a P and
an N cell per magnitude digit), `io_num = col_num / mux_factor = 4`. Each logical
weight is a 3-bit sign-magnitude value (sign + 2 radix-2 magnitude digits) across
4 physical cells (P-MSB, P-LSB, N-MSB, N-LSB). A 2-bit activation drives K=2
serial single-bit WL sub-phases, LSB first.

## Ratios

Both ratio families are derived DOWNWARD from an MSB anchor. Per-digit DSWCT
mirror ratio $r_d = \mathrm{dswct\_ratio\_msb} \cdot \mathrm{radix}^{\,d-(D-1)}$
(LSB-first $d$) → $[0.25, 0.5]$; per-input-bit SINWP-SC leg ratio $s_k =
\mathrm{sc\_ratio\_msb} \cdot 2^{\,k-(K-1)}$ (LSB-first $k$) → $[0.25, 0.5]$. The
composite BL→I_SUB coefficient $r_d \cdot s_k$ over (digit, bit) is $(1/16,\ 1/8,\
1/8,\ 1/4)$, the paper's Fig.7 net transfer.

## Conduction windows

The K input bits run as serial WL sub-phases; sampled bits are held on the
SINWP-SC caps and combined with the live bit at the tail. With K=2 there is one
sample window (bit 0) and the tail. `t_other = t_settle + Σ(ADC step_latency)`.

| Window | Value [ns] | Meaning |
|---|---|---|
| `t_sample[0]` | 2.6 | IN[0] sample sub-phase (held) |
| `t_settle` | 2.66 | tail non-sensing settle |
| `Σ step_latency` | 9.34 | STP1/2/3 = 3.16 / 3.07 / 3.11 (sourced) |
| `t_other` | 12.0 | live/tail = `t_settle + Σ step_latency` |
| `window_array` (per bit) | (2.6, 12.0) | CABLC (whole input branch) / DSWCT conduction |
| `window_sc` (per bit) | (14.6, 12.0) | SINWP-SC held-leg = `Σ t_sample[k:] + t_other` |

## Energy channels → Fig.18 slice mapping

Each rail-to-GND branch is billed at its production site — the macro on its two
named channels (`cablc`, `control`), the readout modules (DSWCT, SINWP-SC,
PN-ISUB), the array (caps), and the TMCSA phase-billing module on their own
module rows; the kernel `SarIadc` is an energy-silent value converter
(`enable_energy_record=false`). The static seats appear in the static PPA
report. Per-slice power at 20 MHz maps to the Fig.18 breakdown as follows,
each slice owned the way the paper's own breakdown draws it:

| Fig.18 slice | Share | Our composition | Billing |
|---|---|---|---|
| Control | 29.2% | `control` channel (`control_config` static seat zero) | 100% dyn `e_control_per_op` (pure per-op) |
| Reference | 23.7% | `reference_config` leakage | 100% static |
| CABLC | 14.9% | `cablc` channel (whole input branch V_DD·I_DL; array caps row folds in) + `cablc_config` leakage | dyn conduction + small/zero static |
| DSWCT | 11.5% | `dswct` module row (self-billed rail conduction) + `dswct_config` seats | dyn conduction + small/zero static |
| SINWP-SC | 8.0% | `sinwp_sc` module row (self-billed held/live legs) + `sinwp_sc_config` seats | dyn conduction + small/zero static |
| PN-ISUB | 3.4% | `pn_isub` module row (3-branch conduction + `e_per_op`) + `pn_isub_config` leakage | dyn conduction + small/zero static |
| TMCSA | 9.3% | `tmcsa` phase-billing module row (per-step PH2/PH3 branch conduction + `e_fixed` per step) + `tmcsa_config` / `adc_config` static seats | dynamic (static seats declared 0) |

The DSWCT mirrors, the SINWP-SC combiner, the PN-ISUB subtractor, and the TMCSA
phase biller are scheme-local reporter modules: each self-bills its rail
conduction on its own profiler row and carries its own (shipped-zero or small)
static seat. The CMD precharge is folded into `e_control_per_op` — no CMD
capacitance is modeled anywhere.

### Comparison caliber (paired slices)

The paper splits ONE series input branch at node V_CMD (the drain of the DSWCT
current-mirror input) between the DSWCT and CABLC slices, and one series sink
branch between SINWP-SC (its sink transistors) and PN-ISUB (switches +
comparator + isub); the internal node voltages are not published, so only the
pair sums are well-defined targets. `validate.py` compares `cablc+dswct`
against 14.9 + 11.5 = 26.4% and `sinwp_sc+pn_isub` against 8.0 + 3.4 = 11.4%,
with `control` / `reference` / `tmcsa` as singles; the four member rows are
reported informationally with no per-member target.

### Cycle normalization (per-access basis)

The paper's 20 MHz power measurement counts **access** cycles: one access = one
mux-slot conversion set (the `io_num` = 4 CIM-IOs in parallel, 16 conducting
columns, `T_AC` = 14.6 ns inside a 50 ns period). A full `vec_mat_mul` over all
`col_num` = 128 logical columns is `mux_factor` = 32 serial accesses. So
`op_frequency` = 20 MHz is the **access rate** and every slice-vs-Fig.18
comparison is on a **per-access** basis: the 641.25 uW per-sub-array budget ⇔
**32.06 pJ/access**. Uniform rule: **every** measured dynamic channel (CABLC,
DSWCT, SINWP-SC, PN-ISUB, TMCSA, and Control alike) is aggregated by the profiler
over one whole VMM, so its per-access energy is the per-VMM total divided by
`mux_factor`. Static leakage is a continuous power and is **never** divided, and
the config seats are the true per-event physical values (Control billed once per
access → `e_control_per_op`; TMCSA `tmcsa_config.e_fixed_per_op__fJ` per
conversion step), not pre-divided. `validate.py` applies this normalization.

## Conventions

- Adopted seats (declared to reproduce a Fig.18 share, NOT fitted to the total):
  Control 29.2% (pure per-op: 100% dynamic `e_control_per_op`, zero static),
  Reference 23.7% (100% static leakage).
- Read path (cablc, dswct, sinwp_sc, pn_isub, tmcsa): pure physics; its static
  seats are declared small/zero and NEVER reverse-solved to fill the total.
- Data: weights value-uniform in [−3, 3], inputs value-uniform in [0, 3] with an
  extra Bernoulli zeroing at `p_zero` (a declared ReLU-sparsity workload assumption).
- Gate: the SINGLE hard gate is total energy per access within ±5% of 32.06
  pJ/access. The paired-slice breakdown is informational — no soft gates.

## Contradiction table

The honest-findings register of the pair-caliber energy campaign — open tensions
the fit could not fully resolve, reported rather than hidden.

| Slice | Open finding |
|---|---|
| TMCSA | The residual budget (74% of its Fig.18 9.3% slice) cannot be absorbed with both the Fig.10(b) as-drawn PH2:PH3 phase occupancy and the `e_fixed_per_op` plausibility ceiling holding at once. The shipped compromise pins `e_fixed_per_op` at its ceiling and widens the phase windows ×1.544 above the as-drawn widths (PH2+PH3 occupancy 74%, vs the as-drawn share). Implication: either the 3×/2× phase-branch conduction model under-counts TMCSA conduction, or the paper's TMCSA Fig.18 slice includes circuitry outside the phase model. See `params.toml` `[cim_macro.tmcsa_config]` for the fitted values and full provenance comment. |

## What is intentionally not modeled

Mismatch and noise (all-off only), the CMD interface capacitance (folded into
the control channel), and a differential ADC (the TMCSA is single-ended).

## Methodology (energy-basis validation)

`validate.py` builds the macro from `params.toml` + `policy.toml`, drives it
directly (the paper 9-row block live) over N random draws per the `anchors.toml`
data conventions, and reduces the profiler to the ENERGY PER ACCESS:
static/access = `leakage_power · t_cycle` (50 ns), dynamic/access = per-VMM
dynamic / `mux_factor`. The single hard gate is the total against 32.06 pJ/access
±5%.

Non-circular rigor: only Control (29.2%) and Reference (23.7%) are `[transcribed]`
seats (Fig.18 shares, for the two peripherals not modeled from physics); the
read-path conduction is pure physics with declared structural constants (g_map,
V_BLC, conduction windows) and static seats declared small/zero, never
reverse-solved; the capacitive / per-op remainders are `[calibrated]` to the
paired-slice residuals inside declared plausibility bounds (see `params.toml`).
`p_zero` is LOCKED to the read-path physics — the value at which the pure-physics
read path conducts its Fig.18 read-path share (47.1% × 32.06 = 15.10 pJ/access),
NOT solved against the total.

Calibration (`tools/calibrate_cell`, `tools/calibrate_solver`, `tools/calibrate.py`;
`[calibrated]` write-back):

0. **Cell chord** — `calibrate_cell` extracts the Linear chord
   (`g_cell_on/off_table`, `vx_ratio`) from the Detail cell at V_BLC = 0.29 V into
   `array_config.cell_config` (g_LRS ≈ 94.6 uS, g_HRS ≈ 5.0 uS; the LRS chord sags
   ~5.4% from the access-NMOS series drop).
1. **Solver** — `calibrate_solver` picks the array DC-solver iteration counts
   (`array_config.solver_config`) by step-ratio plateau on the real wire-R IR-drop
   solve: `n_outer` = 4, `n_inner` = 3.
2. **Ladder re-derivation** — `calibrate.py` probes the unit i_sub staircase (single
   +1 weight, MAC 0..7 over the 9 live rows) and sets `reference_config.i_refs__uA`
   to the adjacent-midpoint taps at this geometry; the ladder TRACKS the calibrated
   chord and verifies code == MAC magnitude (bit-exact staircase).
3. **Adopted seats** — Control / Reference leakage + `e_control_per_op` from the
   Fig.18 shares (declared, not fitted).
4. **p_zero lock** — `calibrate.py` locks `p_zero` where the read path conducts its
   15.10 pJ/access Fig.18 share: `p_zero` = 0.353 (marginal P(x=0) ≈ 0.515).
5. **Cap + per-op remainders** — with conduction FROZEN from a large-N S1
   conduction-row basis, the paired-slice residuals seat the capacitive / per-op
   constants inside declared plausibility bounds: the array wire + cell caps by
   one uniform scale (cablc+dswct pair residual), the SINWP-SC `c_hold`
   (sinwp_sc+pn_isub pair residual minus the kept comparator per-op), and the
   TMCSA `t_ph2`/`t_ph3` + `e_fixed_per_op` joint fit (the 9.3% slice; the
   as-drawn PH2:PH3 ratio kept).

Headline: measured in `results.md` (campaign runs at the LOCKED `p_zero` = 0.353;
marginal P(x=0) ≈ 0.515, independently consistent with typical ~50%-zero post-ReLU
CNN activations). By construction of the calibration above, the two pair slices
and the tmcsa slice close on their Fig.18 shares at the S1 row basis; the hard
gate stays the TOTAL only. The four pair-member rows (cablc, dswct, sinwp_sc,
pn_isub) remain informational with no per-member target (unpublished internal node
voltages); differences are reported, not gated.

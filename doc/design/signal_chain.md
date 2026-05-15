# Signal Chain — OpAmpTIA, ReadOut (bundled), Decoder, ADC

The 1T1R signal chain runs the column current through a non-linear
OpAmpTIA, a **bundled** voltage-domain readout chain (`data S/H +
ref S/H + AnalogMux + differential ADC`), and produces an integer
code.  The two-layer split at the xbar boundary is
`Offset1T1RXbar → Core1T1R` plus `Offset1T1RXbar → ReadOut`:

```
WL: Decoder → DAC → row drive ─┐
                                ↓
SL: Driver → ground bias ──→ Core1T1R (RRAM + NMOS + OpAmpTIA + solver)
                                ↓ v_out_phys (per physical column)
                            (mapping regroup)
                                ↓ (v_data_phys, v_ref_phys)
                              ReadOut  [bundle]
                                ├─ data_switchcap (S/H + WA)
                                ├─ ref_switchcap  (S/H, unit weight)
                                ├─ analog_mux     (differential transport)
                                └─ bl_adc         (differential ADC)
                                          ↓
                                       codes
```

Each leaf component is a stand-alone `nn.Module` with its own
physics-based latency, energy, and PPA model.  The mapping layer
(`Offset1T1RXbar`) is encoding-aware but physics-agnostic; the
physical core is shape-independent and encoding-agnostic.  The new
`ReadOut` ABC (subclass: `OffsetSwitchCapMuxAdcReadOut`) lets the
xbar plug a different readout for a future differential encoding
without rewriting its readout kernel.

## OpAmpTIA (op-amp transimpedance amplifier)

`neurox.analog.opamp_tia.OpAmpTIA` is a non-linear model of the
op-amp + NMOS-pseudo-resistor feedback topology.  The OpAmpTIA owns
its fabricated mismatch state internally and is instantiated
per-`Core1T1R` from a factory (no sharing across cores).  Its API
surface:

- `v_ref__V` (property) — ideal / zero-current clamp voltage.  The
  solver reads it once per `solve` call to seed the Padé warm start.
- `fabricate(shape)` — sample per-column mismatch and **register**
  `self.opamp_gain` internally; delegate to `self.nmos.fabricate(shape)`
  so the pseudo-resistor NMOS submodule fabricates its `β` / `V_th`
  in place.  No return value.
- `snapshot(*, shape: tuple[int, ...]) → OpAmpTIASnapshot` —
  per-VMM dynamic snapshot.  Carries the per-column `opamp_gain` and
  a nested `NMOSSnapshot`; reserved for future runtime noise without
  API churn.
- `solve_dc(i_port__uA, snapshot: OpAmpTIASnapshot, *,
  v_clamp_init__V=None) → OpAmpTIADC` — solve the closed-loop
  equation for the steady-state virtual-ground voltage (`v_clamp`,
  the BL boundary clamp voltage) and rail-limited op-amp output
  voltage (`v_out`, the readout's per-physical-column input).

`OpAmpTIADC` carries `v_clamp__V`, `v_out__V`, `dVclamp_dI__MOhm`,
and `dVout_dI__MOhm`.

Static configuration scalars (`v_ref__V`, `v_nmos_bias__V`,
`v_dd__V`, `output_saturation_softness__V`, `nominal_opamp_gain`)
live on the module as plain attributes / scalar buffers — **not**
registered per-cell buffers — so `torch.compile` folds them as
kernel constants.

### Physical model

The feedback element is an NMOS pseudo-resistor with gate tied to a
fixed bias voltage `v_nmos_bias` (typically VDD).  Drain sits at
the op-amp output `v_out`, source at the BL node `v_clamp`.  The
op-amp's finite open-loop gain `A` couples the two nodes via
`v_out_lin = A · (v_ref − v_clamp)`, with the supply rail modelled
by a **smooth softclip on `v_out_lin`** placed inside the residual
Newton solve (see `temp/opamp_tia.md`):

```
v_out_lin  = A · (v_ref − v_clamp)
v_out      = softclip(v_out_lin; 0, v_dd, s)
            = c + h · tanh((v_out_lin − c) / s)     # c = v_dd/2, h = v_dd/2
f(v_clamp) = nmos.ids(v_g=v_nmos_bias, v_d=v_out, v_s=v_clamp) − i_port
```

`s = output_saturation_softness__V`; defaults to the rail half-span
`v_dd / 2`, giving unit slope at the rail centre (the natural
symmetric-tanh form).  4 unrolled 1D Newton iterations with the
analytical derivative

```
df/dVclamp = ∂I/∂v_d · (−A · g_clip) + ∂I/∂v_s     # g_clip = d softclip/dx
```

After each Newton update `v_clamp` is projected to the physical
range `[0, v_dd]`.  The projection only activates when the upstream
crossbar solver passes an `i_port` mid-convergence that exceeds the
NMOS pseudo-resistor's deliverable current — in that regime the
softclipped residual has no zero and a bare Newton step would
extrapolate wildly.  Stalling at the physical rail lets the outer
solver converge with bounded boundary values; once outer-iteration
currents settle into the NMOS-deliverable range the projection is
inert.

A final eval at the post-projection `v_clamp` recomputes `v_out`,
`g_clip`, and `df/dVclamp` so the returned operating point and the
two sensitivities (`dVclamp_dI__MOhm`, `dVout_dI__MOhm`) all come
from one self-consistent smooth model.  In particular,
`dVout_dI = (−A · g_clip) · dVclamp_dI` naturally rolls toward zero
as `g_clip → 0` in deep saturation — no piecewise derivative mask
is applied.

### Coupling to the 1T1R solver

`NewtonRaphsonSolver1T1R` invokes
`clamp_driver.solve_clamp(i_bl_total, clamp_snapshot, ...)` at the
top of every outer Newton iteration, refreshing the per-column
`v_bl_clamp` from the latest `i_cell.sum(dim=-1)`.  The driver's
small-signal sensitivity `dVclamp_dI__MOhm` (entry 1 of the
returned tuple) is folded into the BL Jacobian via a
Sherman-Morrison rank-1 update — see `solver_1t1r.md` for the
convergence analysis.  `SolverResult` carries the converged
`v_bl_clamp`; `v_out` is obtained by a post-solver re-call to
`opamp_tia.solve_dc(...)` — the OpAmpTIA-specific richer entry
point — with the same `clamp_snapshot`, then handed to the
mapping-layer regroup step.

## ReadOut — bundled chain (S/H + S/H + MUX + ADC)

`neurox.analog.readout.ReadOut` is an ABC for the voltage-domain
readout chain; the offset-coded concrete subclass
`OffsetSwitchCapMuxAdcReadOut` (in
`neurox.analog.readout.offset_switchcap_mux_adc`) owns every block
between the core's per-physical-column `v_out` and the final ADC
code.  It receives **grouped, semantically-labelled** voltages from
the mapping layer — `v_data_grouped: (*R, G, M, D)` and
`v_ref_grouped: (*R, G)` — plus the runtime ADC operating point
`(adc_mode, adc_bits)`, and returns a `ReadOutOutput` dataclass
carrying the integer code, both ADC input voltages, and the
per-block / aggregated energy tensors all keyed by the grouped
lattice `(*R, G, M)`.

Design tenet — leaf modules own all electrical math
---------------------------------------------------
The readout abstraction enforces the project-wide invariant that
**all signal-value changes happen inside leaf circuit modules**;
the readout container only performs shape operations (`unflatten`,
`unsqueeze`, `expand`) and aggregates energies.  The container
must never multiply voltages by a "weight" tensor outside a
`SwitchCap` and must never scale the ref leg by any
encoding-specific geometric constant.  The upper xbar is similarly
constrained: between the core and the readout it may only
`index_select` physical-column outputs into data / ref slots and
`unflatten` the data slot into the grouped lattice.  Both
restrictions ensure the simulated chain matches the silicon chain
block-for-block.

Grouped lattice (xbar-owned regrouping)
---------------------------------------
The xbar is responsible for translating the core's per-physical-
column outputs into the grouped lattice the readout consumes:

* `*P` — physical-instance prefix (`w_phys.shape[:-2]`).
* `G` — number of reference groups per physical array (one ref
  column per group).
* `M` — number of data per group (`ref_group_size`).
* `D` — number of digits per data (`w_digit_count`).

`Offset1T1RXbar._vec_mat_mul_impl` splits `v_out_phys` into data /
ref slots with `index_select`, then `unflatten`s the data slot
into `(G, M, D)`.  The readout receives `v_data_grouped: (*R, G,
M, D)` and `v_ref_grouped: (*R, G)`, so no `data_to_group` lookup
is needed inside the readout — the per-group ref and per-group
data already share the same `G` axis.

Owned submodules:

* **`data_switchcap`** — bottom-plate-sampled cap bank, bank shape
  `(*P, G, M, D)`.  `digit_weights` (shape `[D]`) are broadcast on
  the digit axis at fabricate time; the mapping xbar supplies
  `(w_digit_radix^0, …, w_digit_radix^(D-1))` so the bank's
  averaging weights align with the upstream digit ordering.
  Output is the passive charge-share average
  `Σ_k C_k V_k / Σ_k C_k` per data — a normalised weighted
  average, **not** an unnormalised "digit-weighted sum".
* **`ref_switchcap`** — single-cap bank per reference column, bank
  shape `(*P, G, 1)`, unit ratio.  Models the physical S/H every
  ref column passes through before reaching the ADC's negative
  input.  With one cap per bank the passive charge-share output
  equals the sampled ref voltage itself (within kT/C and per-cap
  mismatch).  The per-data negative leg is built purely by
  `v_neg = v_ref_sampled.unsqueeze(-1).expand(*, G, M)` — no
  algebraic scaling and no index_select.  Rationale: if every
  data digit voltage equals the ref voltage, both legs equal the
  ref voltage and the differential signal is zero, the physically
  correct behaviour.
* **`analog_mux`** — differential transport with gain + CM/DM
  noise + per-access energy.  Fabricated over `(*P, G, 1)` — one
  MUX instance per group with an explicit broadcasting axis for
  the `M` data per group.
* **`bl_adc`** — differential ADC (`convert(v_pos, v_neg, *, mode,
  bits)`).  Default factory: `McsSarAdc` (V_cm-based MCS
  differential SAR).  Fabricated over `(*P, G, 1)` — one ADC per
  group, with the trailing `1` keeping cap-mismatch and
  comparator-offset state broadcast-friendly against the runtime
  `(*R, G, M)` data lattice.  Kept under the historical attribute
  name so the xbar's alias `self.bl_adc = self.readout.bl_adc`
  keeps the macro's `output_rescale_factor` derivation unchanged.

The readout itself is encoding-agnostic at value level: `G`, `M`,
`D` are derived from the `fabricate` arguments (`shape` and
`digit_weights.shape[0]`); the only encoding-specific lookups
(physical-column index tables, digit-weight vector) live on the
mapping xbar.

PPA aggregation
---------------
`OffsetSwitchCapMuxAdcReadOut` aggregates the full chain's PPA so
the upper xbar treats the readout as one opaque block.  Each owned
submodule contributes weighted by its per-array instance count on
the grouped lattice bound by `fabricate(shape, digit_weights)`:

* `data_switchcap` — `group_num * data_num` banks (one bank per data
  column);
* `ref_switchcap` — `group_num` banks (one bank per reference
  column);
* `analog_mux` — `group_num` instances (one MUX per group);
* `bl_adc` — `group_num` instances (one ADC per group).

Concretely:

* `readout.area_per_inst__um2 = cfg.area
   + (group_num * data_num) * data_sc.area
   + group_num * (ref_sc.area + mux.area + bl_adc.area)`.
* `readout.leakage_per_inst__uW` — same multiplicities, for leakage.
* `readout.latency_per_op__ns(*, adc_bits)` is a *method* (not a
  property), because the bound ADC's latency depends on the
  runtime `adc_bits`.  It sums orchestrator + data S/H + ref S/H
  + MUX + `bl_adc.latency_per_op__ns(bits=adc_bits)`.

The PPA properties require `fabricate` to have run first — they
read the lattice counts cached there.

### State holding and per-VMM lifecycle

`SwitchCapConfig` carries only physical knobs (`c_unit__fF`,
matching, `enable_thermal_noise`, PPA); the per-bank *number* and
*relative size* of caps live on the `ratio` tensor passed to
`SwitchCap.fabricate(ratio)`.  Operating temperature `T__K` is an
init kwarg (operating-state, not config), matching the `McsSarAdc`
convention.  `SwitchCap.__init__` registers a 0-d `c__fF` buffer
holding just the unit capacitance — no separate "ideal ladder"
tensor.

`OffsetSwitchCapMuxAdcReadOut.fabricate(shape, digit_weights)`
takes the grouped lattice `(*P, G, M)` and the 1-D digit-weight
vector `[D]` from the upper xbar and fans it out:

* `data_switchcap.fabricate(ratio)` — `ratio.shape == (*P, G, M,
  D)`, built by broadcasting `digit_weights` along the leading
  axes.
* `ref_switchcap.fabricate(ratio)` — `ratio.shape == (*P, G, 1)`
  with unit weights.
* `analog_mux.fabricate((*P, G, 1))` — no-op today; landed for
  lifecycle uniformity with the broadcasting shape used by the
  rest of the chain so future per-leg mismatch has a natural home.
* `bl_adc.fabricate((*P, G, 1))` — one ADC instance per reference
  group; SAR variants size their static cap-mismatch buffers from
  this shape, and the trailing `1` lets the per-group static state
  broadcast cleanly across the `M` data per group at convert time.

Dynamic kT/C sample-and-hold noise is generated inside
`SwitchCap.sample_and_accumulate` via `apply_gaussian` with per-cap
sigma `sqrt(k_B · T / C_k)` against the fabricated `c__fF` (matching
`McsSarAdc` — larger caps see smaller voltage noise).  No per-VMM
snapshot threads through the SwitchCap; neither does the readout
orchestrator.

### Per-VMM energy bookkeeping

`ReadOutOutput.dynamic_energy__fJ` is shape `(*R, G, M)` and is
the sum of:

* `energy_data_sc__fJ` — per-data data-side S/H energy.
* `energy_ref_sc__fJ.unsqueeze(-1).expand(*, G, M) / M` —
  ref-side energy redistributed onto the `M` data columns served
  by each ref bank and divided by `M` so the per-VMM total stays
  conserved (each ref bank fires once per VMM, not once per data).
* `energy_mux__fJ` — per-data MUX access energy.
* `energy_adc__fJ` — per-data ADC conversion energy.
* `ReadOutConfig.energy_per_op__fJ / (G·M)` — orchestrator-level
  share distributed uniformly across data.

The xbar reduces `.sum(dim=(-2, -1))` to obtain the per-VMM total
and flattens the per-data code `(*R, G, M) -> (*R, N=G·M)` for its
returned digital code.

## Differential ADC family

`ADC.convert(v_pos__V, v_neg__V) -> (code, energy)` is the
contract.  Implementations (SAR-MCS, GeneralADC, …) compute on the
differential signal `v_pos__V − v_neg__V`; the SAR family also
keeps the common-mode reference for switched-capacitor sampling.
See [`adc_models.md`](adc_models.md).

## Analog MUX — differential transport

`neurox.analog.analog_mux.AnalogMux` sits inline between readout and
ADC in the offset xbar pipeline.  It is a pure **differential
voltage-transport behavioural block** — no operating-point solve, no
fabricated state, no per-VMM snapshot lifecycle.

`select(v_pos__V, v_neg__V) -> (v_pos_muxed__V, v_neg_muxed__V,
dynamic_energy__fJ)` applies a scalar `mux_gain` to both legs and an
optional inline CM/DM Gaussian transport noise pair (see
`temp/analog_mux.md`):

* `mux_noise_cm_sigma__V` — common-mode sigma.  CM draw is added
  with matching sign on both legs and is suppressed by a differential
  ADC.
* `mux_noise_dm_sigma__V` — differential-mode sigma.  DM draw is
  added to `v_pos` and subtracted from `v_neg`, so it survives a
  differential ADC and directly hurts the signal.

Both sigmas default to `None` (disabled).  Noise tensors are drawn
fresh per `select()` call via `apply_gaussian` — matching the
`SwitchCap` convention (no snapshot threaded from the xbar).

At the default `mux_gain = 1.0` with both sigmas `None` the MUX is a
pure pass-through; the per-access dynamic energy
(`energy_per_access__fJ`) and configured `latency_per_op__ns` still
accrue so chip-level PPA accounting stays consistent.
Latency, leakage and area are plain configured constants — no
pseudo-physical RC derivation, since the project does not track
signal timing at this granularity.

## Decoder

`neurox.analog.decoder.Decoder` wraps the WL DAC with row-decode
and optional bit-serial expansion logic:

* **Non-bit-serial** (default): per-row codes go straight to
  `dac.convert`; the decoder adds row-decode + driver PPA.
* **Bit-serial** (`bit_serial=True`): per-row integer codes are
  expanded into `log2(N)` bit-cycles via `torch.bitwise_and`; the
  macro's Sa loop is unchanged — the bit-cycle axis is inserted as
  a finer cycling dimension below it.

For NeuroX's binary-WL default 1T1R config the decoder operates in
non-bit-serial mode and acts as a pass-through with PPA accounting.

## Memory-mode periphery

Sense amplifiers, write drivers, and precharge logic (used in
memory-mode reads/writes) are intentionally **out of scope**.  The
OpAmpTIA covers the CIM-mode current-to-voltage step; chip-level studies
of memory-mode behaviour need their own primitives.

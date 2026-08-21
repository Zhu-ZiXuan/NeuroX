# xue2020jssc validation -- total energy per access

> **PENDING RECALIBRATION.** Every number below predates the array unification: the scheme-local serial-column array is gone, the macro now runs one `XbarArray1t1r` solve per WL plane over all physical columns, and the capacitive billing moved from the full-cycle `C*V^2` form to the supply-draw law `E = V_rail * C * |dv|` on two declared rails (`v_dd__V`, the new `v_dd_wl__V`). The array/cell capacitance is now stated as one per-node total per line (`bl_node_c__fF`, `x_node_c__fF`, `sl_node_c__fF`, `wl_node_c__fF`), and the boundary driver reaches the first cell across one standard cell pitch. The cap-bearing `[calibrated]` seats in `params.toml` were fitted to the replaced form, so this gate is expected to run RED until the recalibration campaign re-derives them. Codes and currents are unaffected -- the flattening is bit-exact (see the twin test) -- so only the energy rows move.

Energy-basis profiler run for `params.toml` + `policy.toml` on cuda:1. n_w = 64 weight draws x n_x = 256 inputs x 8 rounds = 131072 draws (4194304 accesses), seed 0, run p_zero = 0.348 (marginal P(x=0) = 0.511); anchors-declared workload p_zero = 0.35. Pooled over 8 rounds (relative std of the round totals = 0.22 %).

## Hard gate -- total energy per access

Target 32.06 pJ/access (= 5.13 mW / 8 / 20 MHz); +-5%. At this run's p_zero = 0.348 (marginal P(x=0) = 0.511): result **32.127 pJ/access = 1.002x** +- 0.22 % (round-total relative std over 8 rounds) (err +0.2%), within +-5%: yes.

The anchors-declared workload sparsity is p_zero = 0.35 (marginal P(x=0) ~= 0.511), motivated INDEPENDENTLY by typical ~50%-zero post-ReLU CNN activations -- NOT tuned to pass.

## Energy breakdown (informational -- NOT gated)

| Slice | Energy [pJ/acc] | dyn | static | Fig.18 x 32.06 [pJ] | pred/ref | basis |
|---|--:|--:|--:|--:|--:|:--|
| control |    9.362 |   9.362 |  0.000 |    9.362 |  1.00x | adopted |
| reference |    7.599 |   0.000 |  7.599 |    7.598 |  1.00x | adopted |
| cablc+dswct |    8.510 |   8.510 |  0.000 |    8.464 |  1.01x | physics pair |
| sinwp_sc+pn_isub |    3.671 |   3.671 |  0.000 |    3.655 |  1.00x | physics pair |
| tmcsa |    2.985 |   2.985 |  0.000 |    2.982 |  1.00x | physics |
| cablc |    6.258 |   6.258 |  0.000 |     -    |   -    | pair member |
| dswct |    2.251 |   2.251 |  0.000 |     -    |   -    | pair member |
| sinwp_sc |    1.718 |   1.718 |  0.000 |     -    |   -    | pair member |
| pn_isub |    1.954 |   1.954 |  0.000 |     -    |   -    | pair member |
| **TOTAL (gated)** | **  32.127** |  24.528 |  7.599 | **  32.060** | **1.002x** | PASS (+-5%, err +0.2%) |

The read-path slices are pure physics (g_map, V_BLC, conduction windows -- all declared); control + reference are the two ADOPTED Fig.18 seats. Paired-slice caliber: the paper splits one series input branch at node V_CMD (drain of the DSWCT current-mirror input, Fig.9(a)) between DSWCT and CABLC, and one series sink branch between SINWP-SC (its sink transistors) and PN-ISUB (switches + comparator + isub); the internal node voltages are unpublished, so only the pair sums (cablc+dswct vs 26.4 %, sinwp_sc+pn_isub vs 11.4 %) are well-defined targets -- the member rows are informational. Differences from Fig.18 x 32.06 pJ are reported, not gated.

## Declared conventions

- Hard gate: total energy per access within +-5% of 32.06 pJ at the declared p_zero.
- Adopted seats (declared to reproduce a Fig.18 share, not fitted to the total): control 29.2 % (pure per-op, 100 % dynamic), reference 23.7 % (100 % static).
- Read path (cablc, dswct, sinwp_sc, pn_isub, tmcsa): pure physics; static seats declared small/zero, NEVER reverse-solved to fill the total. Fig.18 comparison at the paired-slice caliber (cablc+dswct, sinwp_sc+pn_isub).
- Data: weights value-uniform in [-3, 3], inputs value-uniform in [0, 3] with an extra Bernoulli zeroing at p_zero (declared workload assumption); rows >= active_row_num zeroed.
- Energy basis: static/access = leakage_power * t_cycle (50 ns); dynamic/access = per-VMM dynamic / mux_factor; total/access = their sum.

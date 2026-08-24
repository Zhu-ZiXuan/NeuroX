# xue2020jssc validation — total energy per access

Energy-basis profiler holdout run for `params.toml` + `policy.toml` on CPU. Calibration used seed 0; this report uses 4 weight draws × 32 inputs × 4 rounds = 512 draws (16,384 accesses), seed 100. The calibrated activity point is P(x≠0) = 0.2935 among at most nine candidate rows and P(w≠0) = 1.0; conditional nonzero magnitudes remain uniform. The relative standard deviation of the four round totals is 2.63%.

## Hard gate — total energy per access

The target is 32,062.5 fJ/access (= 32.0625 pJ = simulated 5.13 mW / 8 / 20 MHz), with ±5% tolerance. The holdout result is **31,959.824 fJ/access = 0.997× target**, an error of **−0.3%**, so the hard gate **passes**.

## Energy breakdown

| Slice | Energy [fJ/access] | Dynamic | Static | Fig.18 × target [fJ] | Pred/ref | Basis |
|---|---:|---:|---:|---:|---:|---|
| Control | 9,362.301 | 6,553.601 | 2,808.700 | 9,362.250 | 1.00× | adopted |
| Reference | 7,599.000 | 0.000 | 7,599.000 | 7,598.812 | 1.00× | adopted |
| CABLC+DSWCT | 8,524.708 | 8,524.708 | 0.000 | 8,464.500 | 1.01× | modeled pair |
| SINWP-SC+PN-ISUB | 3,527.859 | 3,527.859 | 0.000 | 3,655.125 | 0.97× | modeled pair |
| TMCSA | 2,945.956 | 2,945.956 | 0.000 | 2,981.813 | 0.99× | modeled |
| CABLC | 6,200.292 | 6,200.292 | 0.000 | — | — | pair member |
| DSWCT | 2,324.416 | 2,324.416 | 0.000 | — | — | pair member |
| SINWP-SC | 1,503.192 | 1,503.192 | 0.000 | — | — | pair member |
| PN-ISUB | 2,024.666 | 2,024.666 | 0.000 | — | — | pair member |
| **TOTAL (gated)** | **31,959.824** | **21,552.124** | **10,407.700** | **32,062.500** | **0.997×** | **PASS** |

Control and Reference are adopted Fig.18 seats. Control is split into 70% per-access dynamic energy and 30% leakage integrated over the reported 50 ns period; Reference is entirely static. The modeled comparison uses paired slices because CABLC/DSWCT share one series input branch and SINWP-SC/PN-ISUB share one series sink branch, while the paper does not publish the internal node voltages needed to reproduce each member split. The individual member rows therefore show model ownership only and have no separate paper target.

CABLC+DSWCT contains array node-capacitor energy, the complete BL input-branch conduction energy, and DSWCT mirror conduction. The paper-reproduction preset declares the array node capacitances zero because the explicit conduction model already consumes the paired target; this is not a physical extraction. SINWP-SC+PN-ISUB contains SINWP-SC held/live mirror conduction, the three PN-ISUB internal branches, and 47.665 fJ per sign decision and CIM-IO for its three inverters and latch. TMCSA contains its PH2/PH3 branch conduction plus 50 fJ per sensing step and CIM-IO for its internal latch, reset nodes, and local switching. At three bits and four CIM-IOs, that fixed TMCSA term is 600 fJ/access; an external DOUT register is not included.

Static energy integrates leakage over the complete 50 ns period. Dynamic energy is normalized from one VMM over 32 column-MUX accesses. The modeled circuit latency is 14.60 ns and is not used as the static integration window.

The activity probabilities, PN-ISUB event energy, and TMCSA conduction scale were inferred from the same Fig.18 breakdown used here. This result therefore demonstrates internal consistency and holdout-sample stability, not an independent prediction of the published energy.

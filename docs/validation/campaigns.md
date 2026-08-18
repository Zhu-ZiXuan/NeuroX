# Validation campaigns

Evidence that each model in [Reference](../reference/README.md) is both physically faithful and correctly implemented. Where Reference states *what* is modeled, Validation shows *that the implementation reproduces it* — against closed-form limits, SPICE, silicon measurements, or published results.

## Calibration campaigns

The repository's `validations/` directory holds one calibration campaign per published design point, each under its own `validations/<paper>/`. A campaign anchors a scheme's PPA model to that paper's reported numbers and is the citable evidence for that scheme. Each per-paper directory carries a fixed file set:

- `params.toml` — the paper design point (geometry, anchors, window and seat knobs).
- `policy.toml` — the policy the campaign runs under (all-off when the scheme models no non-idealities).
- `anchors.toml` — the reported targets: the hard total, the adopted block shares, and the declared dyn/static and data conventions the calibration assumes.
- `validate.py` and `tools/` — the campaign itself: build from `params.toml`, draw inputs per the declared data conventions, average the profiler per-op energies and read the static report, convert to per-block power, and gate.
- `results.md` — the gate table, the informational per-block breakdown, and every documented miss.

The campaigns that ship today, one per bundled scheme, run as `make validate_<paper>`. They live outside `docs/`, so this page names their paths instead of linking them:

- `validations/xue2020jssc/` — the SINWP 1T1R CIM sub-array of `neurox.works.macro.cim.xue2020jssc`; `README.md` carries the run instructions and the campaign's contradiction table, `results.md` the recorded outcome.
- `validations/ye2023jssc/` — the WH-2T1R CIM macro of `neurox.works.macro.cim.ye2023jssc`; the `validate.py` module docstring carries the run instructions and the gate list, `results.md` the recorded outcome.

Four conventions govern every campaign:

- **The hard gate is the headline total only.** One tight-tolerance gate on the total energy per access (in the scheme's energy basis) is the campaign's pass/fail. The per-block breakdown is reported alongside it but is informational — a share differing from the paper's is explained (a convention or node-voltage difference), not gated, since the paper defines no per-block boundaries.
- **Adopt versus predict is declared per seat.** A seat whose value comes from the paper's reported number (a block share, a measured total) is *adopted* and fixed; the rest of the model is *predicted* from physics under declared assumptions. The read path is pure physics — no residual-absorbing leakage is fitted to close the gap — and the adopted-versus-predicted split is stated in `anchors.toml`.
- **Assumed distributions are declared and swept, never fitted.** An input-statistics assumption the paper does not pin down — an activation sparsity `p_zero`, for instance — is a declared value plus a sensitivity sweep across its plausible range; the headline claim is that the total brackets the target across that range. Such a knob is never tuned to land the gate.
- **Misses are documented, never tuned away.** A gate miss is recorded in `results.md` with its diagnosis and the contingency step taken, following the campaign's declared contingency order. A seat is never silently adjusted to pass, and a suspicious deviation — a clean factor-of-two or factor-of-four — is called out as a suspected unstated model term before any knob is moved.

### Provenance tags

Every key of `params.toml` and `anchors.toml` that names a physical quantity carries a provenance tag, stated in its own inline comment or in the comment block directly above it, so a reader can separate what the paper reports from what the model assumes or solves. Keys that name no physical quantity are exempt: registry dispatch keys (`_neurox_class`) and pure selectors. This table is the authoritative legend:

| Tag | Meaning |
|---|---|
| `[measured pN]` | read off a paper figure or table on page N |
| `[derived]` | computed from tagged values by the stated arithmetic |
| `[transcribed]` | a measured paper block power adopted verbatim as a constant |
| `[assumed]` | a prior or campaign convention, not paper-sourced |
| `[bound-derived]` | set so the model saturates a measured inequality |
| `[calibrated]` | solved against a stated validation constraint |

These value-level campaign tags are distinct from the module-parameter **Source** classes of [module parameter](../conventions/module_parameter.md), which classify a parameter of a documented model rather than a value of a shipped config artifact.

## TODO

Per-subsystem validation notes cross-linked from each Reference document's *Validation* section, solver verification (converged-residual, per-cell finite-difference device-derivative, and chunking bit-exactness checks), and cross-tool / cross-simulator comparison.

# Validation

Evidence that each model in [Reference](../reference/README.md) is both physically faithful and correctly implemented. Where Reference states *what* is modeled, Validation shows *that the implementation reproduces it* — against closed-form limits, SPICE, silicon measurements, or published results.

Planned contents (one entry per claim worth defending):

- per-subsystem validation notes, cross-linked from each Reference document's *Validation* section;
- solver verification (e.g. converged-residual checks, per-cell finite-difference device-derivative checks, chunking bit-exactness);
- cross-tool / cross-simulator comparison.

This section is largely a set of placeholders today; entries are filled as comparison data and tests accumulate.

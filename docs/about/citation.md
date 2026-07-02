# Citation and versioning

## How to cite NeuroX

If you use NeuroX in academic work, please cite it. Machine-readable citation metadata is kept in `CITATION.cff` at the repository root, which GitHub renders into a "Cite this repository" entry.

TODO — fill in `CITATION.cff` (authors, title, year, DOI/URL) and mirror a BibTeX entry here once the canonical reference exists.

```bibtex
TODO — BibTeX entry (add once CITATION.cff is populated)
```

## Public API surface

The supported public surface of NeuroX stops at `neurox.macro`. The device, crossbar, analog, digital, and profiler layers below it are implementation detail: their classes, configs, and signatures may change without a deprecation cycle. Build against `neurox.macro` for any code you intend to keep working across releases; reach below it only with the understanding that it is unstable by design.

## Versioning policy

NeuroX is at version 0.1.0. While the project is pre-1.0, any release may change the API, including the `neurox.macro` surface.

TODO — state the post-1.0 stability and deprecation policy (semantic-versioning contract, what a major/minor/patch bump guarantees for `neurox.macro`, deprecation window) once the public API is frozen for a 1.0 release.

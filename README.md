# NeuroX

NeuroX is a PyTorch-based simulator for RRAM compute-in-memory hardware. It connects device and circuit models to crossbar arrays, CIM macros, and integer linear and convolution operators. Profiling reports modeled power, performance, and area (PPA) quantities alongside numerical outputs.

## Start using the library

From a source checkout, install NeuroX into an activated environment with a compatible PyTorch installation:

```bash
python -m pip install .
```

Follow the [quickstart](docs/get_started/README.md) to program and execute a small integer operator and inspect its profile. It runs on CPU without model checkpoints or datasets. Supported dependency ranges are declared in `pyproject.toml`.

## Documentation

- [Usage guides](docs/guides/README.md) — configuration, execution, profiling, and calibration.
- [API reference](docs/api/README.md) — application interfaces, component families, and extension contracts generated from source.
- [Scientific models](docs/reference/README.md) — equations, assumptions, parameter provenance, and validity.
- [Validation](docs/validation/README.md) — methods and runnable paper-specific campaigns.
- [System explanations](docs/system_design/README.md) — physical state, execution, and PPA interpretation.
- [Scope and limitations](docs/about/scope_limitations.md) — what to account for when interpreting results.
- [Contributing](CONTRIBUTING.md) — prepare a checkout and submit a change.

NeuroX is released under the [MIT License](LICENSE).

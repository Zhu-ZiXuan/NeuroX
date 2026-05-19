# Operator Training Path

This directory documents the current training-side operator path.

Current status:

- runtime inference operators live under `neurox/operator/linear/` and `neurox/operator/conv/`
- the HAT/QAT recipe lives under `neurox/operator/train/`
- observer and fake-quant utilities are reusable training primitives
- `HATLinear` / `HATConv2d` remain one concrete training/export recipe rather than the only possible training path

The current training path exists to produce NeuroX-flat integer state and to exercise hardware-aware training against the macro backend. It should not be confused with the runtime operator layer.

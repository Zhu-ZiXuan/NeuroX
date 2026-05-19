# Operator Modules

This directory documents the operator layer — the bridge between the PyTorch `nn.Module` surface user code interacts with and the macro backend that actually performs the integer-domain VMM.

## Layout

`neurox/operator/` is split into:

- `base.py` — `NeuroxOperator` abstract base for quantised operators.
- `spec.py` — `QuantSpec` carrying the integer-quantisation grid every operator agrees on (`x_qmin`, `x_qmax`, `w_qmax`, `y_qmin`, `y_qmax`).
- `linear/`, `conv/` — concrete operator subclasses for `nn.Linear` / `nn.Conv2d` replacements (`QuantLinear`, `QuantConv2d` for inference; `HATLinear`, `HATConv2d` for hardware-aware training).
- `train/` — training utilities (`HAT` recipe, fake-quant kernels, observers, BN-fold helpers).
- `qat_util.py` — small QAT helpers shared across the inference and training operator paths.

## Why this layer exists

Operators are the only NeuroX surface that PyTorch user code talks to directly. Every per-layer macro instance lives inside one operator. The operator layer therefore owns:

- the integer-domain quantisation grid (`QuantSpec`)
- the per-call broadcast between PyTorch tensor shapes and the macro's expected `(M, Tc, Tr, Sa, Sw)` axes
- the straight-through-estimator and fake-quant gradient paths used by the training recipe

It does **not** own analog physics, mapping, or aggregation logic — those belong to the macro / mapper / xbar layers below.

## Training vs. inference

- Inference: `QuantLinear` / `QuantConv2d` consume integer weights / activations and call into the macro for the per-cycle VMM.
- Training: `HATLinear` / `HATConv2d` add fake-quant kernels, EMA-based activation range tracking, and the STE gradient path so the macro backend can be exercised under autograd. See [`docs/dev/modules/operator/train/README.md`](docs/dev/modules/operator/train/README.md).

See also:

- `docs/dev/modules/macro/README.md`
- `docs/dev/modules/replace/README.md`
- `docs/dev/modules/operator/train/README.md`

# Replace Modules

`neurox/replace/` owns the model-rewriting pipeline that swaps stock PyTorch modules (`nn.Linear`, `nn.Conv2d`, …) for NeuroX quantised / HAT-equipped counterparts.

## Public surface

- `replace.replace_for_hat(...)` — walk a `nn.Module`, replace eligible leaf modules with their NeuroX HAT counterparts (`HATLinear`, `HATConv2d`).
- `policy.py` — the replacement-policy abstractions (which modules to replace, which to leave untouched).
- `state.py` — the per-replacement state container (NeuroX layer name prefix, macro factory closure, …).

## When to use

The replacement pipeline is the production way to turn an existing PyTorch model into a NeuroX-trainable / NeuroX-evaluable model. Tests and example scripts call it directly; user-level code typically goes through the `examples.md` entry points.

## Why this is its own package

Model rewriting is a workflow concern, not a per-layer concern. Keeping it under its own package isolates the FX / tree-walk logic from the per-module physics. Future plans (`torch.fx`-based automatic graph rewriting, per-layer fine-grained policy) will land here.

See also:

- `docs/dev/modules/macro/README.md`
- `docs/dev/modules/operator/README.md`
- `docs/dev/roadmap.md`

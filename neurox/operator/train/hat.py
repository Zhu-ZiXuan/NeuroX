"""HAT pipeline helpers: replace, freeze observers, extract state.

See also:
    docs/dev/modules/operator/train/README.md
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.operator.spec import QuantSpec

from .observer import PerChannelSymmObserver, PerTensorObserver


def replace_for_hat(
    model: nn.Module,
    macro_factory: Callable[..., NeuroxMacroQuantMatMul],
    spec: QuantSpec,
    *,
    exclude: set[str] | None = None,
) -> nn.Module:
    """Swap every ``nn.Linear`` / ``nn.Conv2d`` for its HAT counterpart.

    Args:
        model: Float model (already BN-folded if applicable).
        macro_factory: Callable returning a fresh macro per layer;
            receives the qualified module name as ``name=`` kwarg.
        spec: Quantization grid applied to every replaced layer.
        exclude: Optional set of qualified module names to skip.

    Returns:
        The same model object, mutated in place.
    """
    from neurox.operator.conv import HATConv2d, QuantConv2d
    from neurox.operator.linear import HATLinear, QuantLinear

    skip = exclude or set()

    def recurse(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if qualified in skip:
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, (QuantLinear, HATLinear)):
                setattr(module, name, HATLinear.from_torch(child, macro_factory(name=qualified), spec, qualified))
            elif isinstance(child, nn.Conv2d) and not isinstance(child, (QuantConv2d, HATConv2d)):
                setattr(module, name, HATConv2d.from_torch(child, macro_factory(name=qualified), spec, qualified))
            else:
                recurse(child, qualified)

    recurse(model, "")
    return model


def freeze_hat_observers(model: nn.Module) -> int:
    """Pin every HAT observer's ``(min, max)`` / ``abs_max`` stats.

    Typical HAT schedule: run a calibration pass (train-mode + no-grad)
    so observers settle their running stats, then call this to lock
    them in place for the rest of training.  Locking avoids a known
    positive-feedback loop — EMA updates during training chase the
    STE-noisy output ranges, which re-scales ``(s_y, zp_y)`` between
    steps and destabilises the integer requantize pipeline.

    Returns:
        Number of observers that were actually frozen (for logging).
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (PerTensorObserver, PerChannelSymmObserver)):
            module.freeze()
            count += 1
    return count


def extract_neurox_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect a ``neurox_flat`` state_dict from every HAT op in ``model``.

    Walks the model, calls each :class:`HATLinear` / :class:`HATConv2d`'s
    ``extract_neurox_state`` with its qualified module name, and
    stitches the 12-buffer per-layer dicts together.  Non-HAT state
    (BN folded to Identity, float ReLU/MaxPool, etc.) is copied
    through unchanged so the returned dict is a complete state_dict
    for a model that has been re-replaced with :class:`QuantLinear` /
    :class:`QuantConv2d` at eval time.
    """
    from neurox.operator.conv import HATConv2d
    from neurox.operator.linear import HATLinear

    state: dict[str, torch.Tensor] = {}
    quantized_prefixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, (HATLinear, HATConv2d)):
            state.update(module.extract_neurox_state(name))
            quantized_prefixes.add(f"{name}.")
    for key, value in model.state_dict().items():
        if any(key.startswith(p) for p in quantized_prefixes):
            continue
        state[key] = value.detach().cpu().clone()
    return state

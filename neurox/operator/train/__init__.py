"""Training-only utilities for hardware-aware training.

This subpackage hosts the components that exist only on the HAT
training path:

- :class:`PerTensorObserver` / :class:`PerChannelSymmObserver` — EMA
  min/max observers used by the HAT operators.
- :func:`fake_quant_ste` / :func:`fake_quant_symm_per_channel_ste` —
  straight-through estimator fake-quant helpers.
- :func:`fold_batchnorm` — pre-HAT BN folding into the preceding
  Conv / Linear.
- :func:`replace_for_hat` / :func:`freeze_hat_observers` /
  :func:`extract_neurox_state` — the high-level pipeline functions.

The HAT operator *classes* themselves (:class:`HATLinear` /
:class:`HATConv2d`) live alongside their inference counterparts in
:mod:`neurox.operator.linear` and :mod:`neurox.operator.conv`.
"""

from .fake_quant import fake_quant_ste, fake_quant_symm_per_channel_ste
from .fold import fold_batchnorm
from .hat import extract_neurox_state, freeze_hat_observers, replace_for_hat
from .observer import PerChannelSymmObserver, PerTensorObserver

__all__ = [
    "PerChannelSymmObserver",
    "PerTensorObserver",
    "extract_neurox_state",
    "fake_quant_ste",
    "fake_quant_symm_per_channel_ste",
    "fold_batchnorm",
    "freeze_hat_observers",
    "replace_for_hat",
]

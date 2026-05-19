"""Training-only utilities for hardware-aware training.

See also:
    docs/dev/modules/operator/train/README.md
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

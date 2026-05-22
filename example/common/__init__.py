"""Shared helpers for NeuroX example scripts.

- ``macro_factory``: :func:`build_macro_factory` and
  :func:`derive_quant_spec` — the two pieces every example needs to
  wire its macro TOML and ``--xbar`` choice into
  ``neurox.replace_for_hat`` / ``neurox.build_evaluator``.
- ``pt2e``: one-way extractor from a torchao pt2e prepared graph into
  the NeuroX-flat state_dict schema.
- ``procedures``: shared evaluation loop used by every example's
  ``evaluate.py``.
"""

from .macro_factory import XbarKind, build_macro_factory, derive_quant_spec
from .procedures import run_evaluate
from .pt2e import pt2e_to_neurox_state

__all__ = [
    "XbarKind",
    "build_macro_factory",
    "derive_quant_spec",
    "pt2e_to_neurox_state",
    "run_evaluate",
]

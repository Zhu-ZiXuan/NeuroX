"""Quantization-grid specification shared by the HAT operators."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSpec:
    """Uniform quantization grid for one HAT layer.

    The grids are the operator-side contract; they must fit the target
    macro's published ``x_value_range`` / ``w_value_range``.  The
    matching check runs at HAT construction (see
    ``NeuroxOperator.validate_spec_against_macro``).

    Attributes:
        x_qmin: Activation integer minimum (e.g. ``0`` for a 4-bit
            unsigned grid).  Must satisfy
            ``macro.x_value_range[0] <= x_qmin``.
        x_qmax: Activation integer maximum.  Must satisfy
            ``x_qmax <= macro.x_value_range[1]``.
        w_qmax: Symmetric weight bound; integer range becomes
            ``[-w_qmax, +w_qmax]``.  Must fit
            ``macro.w_value_range``, e.g. ``w_qmax=63`` when the chosen
            macro publishes ``w_value_range=(-63, 63)``.
        y_qmin: Output integer minimum.  Operator-side boundary between
            layers; not bound to the macro's input grid.
        y_qmax: Output integer maximum.

    The ``x`` and ``y`` grids apply asymmetric affine quantization
    (scale + zero-point); ``w`` uses symmetric per-output-channel.
    """

    x_qmin: int = 0
    x_qmax: int = 15
    w_qmax: int = 63
    y_qmin: int = 0
    y_qmax: int = 15

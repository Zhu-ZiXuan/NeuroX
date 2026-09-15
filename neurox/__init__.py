"""Public API for constructing and running NeuroX compute units."""

from . import architecture, primitive, works
from .api.factory import (
    cim_macro_from_file,
    conv2d_unit_from_file,
    diff_vadc_from_file,
    iadc_from_file,
    linear_unit_from_file,
)
from .api.function import check_unique_binding, fabricate, set_temperature, stamp_names
from .api.profiler import Profiler
from .api.reporter import Reporter

__all__ = [
    "architecture",
    "primitive",
    "works",
    "Profiler",
    "Reporter",
    # function
    "check_unique_binding",
    "fabricate",
    "set_temperature",
    "stamp_names",
    # factory
    "cim_macro_from_file",
    "linear_unit_from_file",
    "conv2d_unit_from_file",
    "diff_vadc_from_file",
    "iadc_from_file",
]

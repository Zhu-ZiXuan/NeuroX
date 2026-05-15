"""Common utilities for NeuroX.

Importing this package also installs the project-wide ``torch._dynamo``
configuration.  ``neurox.common`` is a transitive dependency of every
other ``neurox.*`` sub-package, so the bootstrap fires exactly once on
the first ``from neurox.<sub> import ...`` regardless of entry point.
"""

from .config import Config, ConfigBuilder, ValidationError
from .load_dump import (
    dataclass_from_dict,
    dataclass_from_file,
    dataclass_to_dict,
    dataclass_to_file,
    dict_configs_from_file,
    dict_configs_to_file,
    dict_from_file,
    dict_to_file,
    merge_dicts,
)
from .physical_constant import (
    ELEM_CHARGE__C,
    T_ROOM__K,
    EPS_0__F_per_m,
    K_BOLTZMANN__J_per_K,
    thermal_voltage__V,
)
from .quant import (
    floor_bucketize,
    stochastic_floor_div,
    stochastic_floor_to_int,
    use_stochastic,
)

__all__ = [
    "Config",
    "ConfigBuilder",
    "ELEM_CHARGE__C",
    "EPS_0__F_per_m",
    "K_BOLTZMANN__J_per_K",
    "T_ROOM__K",
    "ValidationError",
    "dataclass_from_dict",
    "dataclass_from_file",
    "dataclass_to_dict",
    "dataclass_to_file",
    "dict_configs_from_file",
    "dict_configs_to_file",
    "dict_from_file",
    "dict_to_file",
    "floor_bucketize",
    "merge_dicts",
    "stochastic_floor_div",
    "stochastic_floor_to_int",
    "thermal_voltage__V",
    "use_stochastic",
]

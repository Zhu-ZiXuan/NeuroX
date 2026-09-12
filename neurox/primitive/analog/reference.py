"""Fabricated reference tensor with caller-owned axis and unit semantics.

See Also:
    docs/reference/primitive/analog/reference.md
    docs/system_design/physical_state.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import apply_relative_gaussian

type FloatArray = float | tuple[FloatArray, ...]


def _array_shape(value: FloatArray, *, path: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        return ()
    if not value:
        raise ValueError(f"require: len({path}) (0) >= 1")

    item_shape = _array_shape(value[0], path=f"{path}[0]")
    for index, item in enumerate(value[1:], start=1):
        shape = _array_shape(item, path=f"{path}[{index}]")
        if shape != item_shape:
            raise ValueError(
                f"require: {path} rectangular; {path}[{index}] shape {shape} == {path}[0] shape {item_shape}"
            )
    return (len(value), *item_shape)


class ReferenceConfig(ConfigBase):
    values: FloatArray
    """Nominal scalar or rectangular tuple tree. Axis meanings and units belong
    to the owner."""
    tolerance_sigma_relative: float
    """Relative per-instance initial-accuracy sigma, sampled at fabrication."""
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    """All standing power of the physical reference generator."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the configured value array."""
        return _array_shape(self.values, path="values")

    def validate(self) -> None:
        _array_shape(self.values, path="values")
        self._require_non_neg(self.tolerance_sigma_relative, "tolerance_sigma_relative")
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


class ReferencePolicy(PolicyBase):
    tolerance: bool
    """Apply `tolerance_sigma_relative` at fabrication."""


_Config = ReferenceConfig
_Policy = ReferencePolicy


class Reference(ModuleBase):
    """Return one fabricated tensor without interpreting its axes or units."""

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_values: Tensor  # Shape: [...]

    # === Fabricated state ===

    _values: Tensor  # Shape: [*inst_shape, ...]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        del T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_nonpersistent_buffer("_nominal_values", torch.tensor(config.values, dtype=dtype))

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def _sample_fabrication_variation(self) -> None:
        values = self._nominal_values.clone().expand((*self.inst_shape, *self.config.shape))
        self._values = apply_relative_gaussian(
            values,
            self.config.tolerance_sigma_relative,
            enabled=self.policy.tolerance,
        )

    @torch.no_grad()
    def values(self) -> Tensor:
        """Return the fabricated tensor unchanged."""
        return self._values

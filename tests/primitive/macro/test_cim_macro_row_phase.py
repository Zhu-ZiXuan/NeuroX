"""Row activation geometry on the CimMacro base.

- ``CimMacroConfig`` geometry guards for ``active_row_num`` (range only; the
  base imposes no ``row_num % active_row_num`` divisor — a non-divisible tile
  is valid, the engine covers every row with a short final sub-phase block, and
  uniform row-blocking is an operator-layer contract, see
  ``tests/architecture/unit/test_linear_cim_unit.py``).
- ``max_active_num``: the single sub-phase query for upper layers, reading
  ``config.active_row_num``.
- ``_split_col_lanes``: trailing col axis -> ``(lane_num, col_per_lane)``
  with ``lane = col // col_per_lane``; exact divisibility required.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
import torch

from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


class _IdealCimMacroKwargs(TypedDict):
    col_num: int
    row_num: int
    active_row_num: int
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    x_value_range: tuple[int, int]
    w_digit_count: int
    w_digit_radix: int
    w_digit_value_range: tuple[int, int]
    adc_mode_num: int
    adc_max_bits: int


def _config_kwargs(*, row_num: int, active_row_num: int) -> _IdealCimMacroKwargs:
    return {
        "col_num": 4,
        "row_num": row_num,
        "active_row_num": active_row_num,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "x_value_range": (0, 1),
        "w_digit_count": 1,
        "w_digit_radix": 2,
        "w_digit_value_range": (0, 1),
        "adc_mode_num": 1,
        "adc_max_bits": 8,
    }


def _make_xbar(*, row_num: int, active_row_num: int, inst_shape: tuple[int, ...] = ()) -> IdealCimMacro:
    config = IdealCimMacroConfig(**_config_kwargs(row_num=row_num, active_row_num=active_row_num))
    xbar = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    xbar.eval()
    return xbar


# ---------------------------------------------------------------------------
# active_row_num geometry validation
# ---------------------------------------------------------------------------


class TestActiveRowNumValidation:
    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"active_row_num"):
            IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=0))

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"active_row_num"):
            IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=-2))

    def test_above_row_num_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"active_row_num"):
            IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=16))

    def test_non_divisor_accepted(self) -> None:
        # The base imposes no row_num % active_row_num divisor; a non-divisible
        # tile is a valid macro (the engine tiles it with a short final block).
        # Uniform row-blocking is enforced one layer up, at LinearCimUnit.
        cfg = IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=3))
        assert cfg.active_row_num == 3

    def test_full_activation_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=8))
        assert cfg.active_row_num == 8

    def test_max_active_num_property(self) -> None:
        xbar = _make_xbar(row_num=8, active_row_num=2)
        assert xbar.max_active_num == 2
        full = _make_xbar(row_num=8, active_row_num=8)
        assert full.max_active_num == 8


# ---------------------------------------------------------------------------
# _split_col_lanes
# ---------------------------------------------------------------------------


class TestSplitColLanes:
    def test_lane_order_and_mapping(self) -> None:
        t = torch.arange(8)
        y = CimMacro._split_col_lanes(t, col_per_lane=4)
        assert y.shape == (2, 4)
        # lane = col // col_per_lane: lane axis first, serial position within
        # the lane trailing.
        for col in range(8):
            assert y[col // 4, col % 4].item() == col

    def test_batch_prefix_preserved(self) -> None:
        t = torch.randn(3, 5, 8)
        y = CimMacro._split_col_lanes(t, col_per_lane=2)
        assert y.shape == (3, 5, 4, 2)
        assert torch.equal(y.flatten(start_dim=-2), t)

    def test_non_divisible_rejected(self) -> None:
        t = torch.arange(10)
        with pytest.raises(ValueError, match=r"col_per_lane"):
            CimMacro._split_col_lanes(t, col_per_lane=4)

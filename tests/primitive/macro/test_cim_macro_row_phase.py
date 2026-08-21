"""Input activation geometry on the CimMacro base.

- `CimMacroConfig` geometry guards for `max_active_num`.
- `max_active_num` exposes the per-conversion selection limit.
- `_split_col_lanes`: splits the trailing col axis into a lane grid with
  `lane = col // col_per_lane`; exact divisibility required.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
import torch

from neurox.primitive.macro.cim import CimMacro, IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


class _IdealCimMacroKwargs(TypedDict):
    max_active_num: int
    area_per_inst__um2: float
    leakage_per_inst__uW: float
    x_value_range: tuple[int, int]
    w_value_range: tuple[int, int]
    quantization_input_ranges: tuple[tuple[int, int], ...]
    adc_max_bits: int


def _config_kwargs(*, max_active_num: int) -> _IdealCimMacroKwargs:
    return {
        "max_active_num": max_active_num,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "x_value_range": (0, 1),
        "w_value_range": (0, 1),
        "quantization_input_ranges": ((-8, 7),),
        "adc_max_bits": 8,
    }


def _make_macro(*, input_num: int, max_active_num: int, inst_shape: tuple[int, ...] = ()) -> IdealCimMacro:
    config = IdealCimMacroConfig(**_config_kwargs(max_active_num=max_active_num))
    macro = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        input_num=input_num,
        output_num=4,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )
    macro.eval()
    return macro


# ---------------------------------------------------------------------------
# max_active_num geometry validation
# ---------------------------------------------------------------------------


class TestMaxActiveSizeValidation:
    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=0))

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            IdealCimMacroConfig(**_config_kwargs(max_active_num=-2))

    def test_above_input_num_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"max_active_num"):
            _make_macro(input_num=8, max_active_num=16)

    def test_non_divisor_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(max_active_num=3))
        assert cfg.max_active_num == 3

    def test_full_activation_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(max_active_num=8))
        assert cfg.max_active_num == 8

    def test_max_active_num_property(self) -> None:
        macro = _make_macro(input_num=8, max_active_num=2)
        assert macro.max_active_num == 2
        full = _make_macro(input_num=8, max_active_num=8)
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

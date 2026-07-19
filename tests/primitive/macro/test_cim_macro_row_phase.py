"""Row active-phase serialization mechanics on the CimMacro base.

- ``CimMacroConfig`` geometry guards for ``active_row_num`` (range and
  divisibility) and the derived ``active_phase_num``.
- ``_active_row_mask`` / ``_unroll_row_phase``: phase ``p`` keeps exactly
  rows ``[p*A, (p+1)*A)``, zeros elsewhere, dtype preserved.
- ``_split_col_lanes``: trailing col axis -> ``(lane_num, col_per_lane)``
  with ``lane = col // col_per_lane``; exact divisibility required.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.macro.cim import CimMacro
from neurox.primitive.macro.cim.ideal import IdealCimMacro, IdealCimMacroConfig, IdealCimMacroPolicy


def _config_kwargs(*, row_num: int, active_row_num: int) -> dict[str, object]:
    return {
        "col_num": 4,
        "row_num": row_num,
        "active_row_num": active_row_num,
        "area_per_inst__um2": 0.0,
        "leakage_per_inst__uW": 0.0,
        "x_range": (0, 1),
        "w_digit_count": 1,
        "w_digit_radix": 2,
        "w_digit_range": (0, 1),
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

    def test_non_divisor_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"active_row_num"):
            IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=3))

    def test_full_activation_accepted(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=8))
        assert cfg.active_phase_num == 1

    def test_active_phase_num_derived(self) -> None:
        cfg = IdealCimMacroConfig(**_config_kwargs(row_num=8, active_row_num=2))
        assert cfg.active_phase_num == 4


# ---------------------------------------------------------------------------
# _active_row_mask / _unroll_row_phase
# ---------------------------------------------------------------------------


class TestUnrollRowPhase:
    def test_mask_partitions_rows(self) -> None:
        xbar = _make_xbar(row_num=8, active_row_num=2)
        mask = xbar._active_row_mask
        assert mask.shape == (4, 8)
        assert mask.dtype == torch.bool
        rows = torch.arange(8)
        expected = rows.unsqueeze(0) // 2 == torch.arange(4).unsqueeze(-1)
        assert torch.equal(mask, expected)
        # Each row belongs to exactly one phase.
        assert torch.equal(mask.sum(dim=0), torch.ones(8, dtype=mask.sum(dim=0).dtype))

    def test_unroll_keeps_own_rows_only(self) -> None:
        xbar = _make_xbar(row_num=8, active_row_num=2)
        x = torch.arange(1, 9, dtype=torch.int32)
        y = xbar._unroll_row_phase(x)
        assert y.shape == (4, 8)
        assert y.dtype == torch.int32
        for p in range(4):
            own = slice(p * 2, (p + 1) * 2)
            assert torch.equal(y[p, own], x[own])
            others = y[p].clone()
            others[own] = 0
            assert others.abs().sum().item() == 0

    def test_unroll_preserves_batch_prefix_and_float_dtype(self) -> None:
        xbar = _make_xbar(row_num=8, active_row_num=4)
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        y = xbar._unroll_row_phase(x)
        assert y.shape == (2, 3, 2, 8)
        assert y.dtype == torch.float32
        expected = torch.where(xbar._active_row_mask, x.unsqueeze(-2), x.new_zeros(()))
        assert torch.equal(y, expected)

    def test_single_phase_axis_present(self) -> None:
        xbar = _make_xbar(row_num=8, active_row_num=8)
        x = torch.arange(1, 9, dtype=torch.int32)
        y = xbar._unroll_row_phase(x)
        assert y.shape == (1, 8)
        assert torch.equal(y[0], x)

    def test_phase_axis_left_of_inst_span(self) -> None:
        """Non-empty inst prefix: the phase axis inserts LEFT of the span."""
        xbar = _make_xbar(row_num=8, active_row_num=2, inst_shape=(3,))
        x = torch.randn(3, 8)  # inst-alignment axis present in x
        y = xbar._unroll_row_phase(x)
        assert y.shape == (4, 3, 8)
        for p in range(4):
            expected = torch.zeros_like(x)
            expected[:, p * 2 : (p + 1) * 2] = x[:, p * 2 : (p + 1) * 2]
            assert torch.equal(y[p], expected)

    def test_inst_span_singletons_when_x_omits_it(self) -> None:
        """x without inst axes gains size-1 slots so alignment cannot collide."""
        xbar = _make_xbar(row_num=8, active_row_num=4, inst_shape=(2, 3))
        x = torch.arange(1, 9, dtype=torch.int32)
        y = xbar._unroll_row_phase(x)
        assert y.shape == (2, 1, 1, 8)
        assert y.dtype == torch.int32
        assert torch.equal(y[0, 0, 0, :4], x[:4])
        assert torch.equal(y[1, 0, 0, 4:], x[4:])
        assert y[0, 0, 0, 4:].abs().sum().item() == 0
        assert y[1, 0, 0, :4].abs().sum().item() == 0

    def test_batch_rides_left_of_phase_with_inst_span(self) -> None:
        """Batch leading stays left of the phase axis; inst span stays right."""
        xbar = _make_xbar(row_num=8, active_row_num=4, inst_shape=(3,))
        x = torch.randn(5, 3, 8)  # (batch, inst, row)
        y = xbar._unroll_row_phase(x)
        assert y.shape == (5, 2, 3, 8)
        for p in range(2):
            expected = torch.zeros_like(x)
            expected[..., p * 4 : (p + 1) * 4] = x[..., p * 4 : (p + 1) * 4]
            assert torch.equal(y[:, p], expected)


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

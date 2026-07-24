"""Tests for the engine-level sub-phase plane machinery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from neurox.architecture.unit.cim.engine import (
    CimEngine,
    CimEnginePolicy,
    DirectCimEngine,
    DirectCimEngineConfig,
    DirectCimEnginePolicy,
    InterArraySliceCimEngine,
    InterArraySliceCimEngineConfig,
    InterArraySliceCimEnginePolicy,
    IntraArraySliceCimEngine,
    IntraArraySliceCimEngineConfig,
    IntraArraySliceCimEnginePolicy,
)
from neurox.common.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import IdealCimMacroConfig, IdealCimMacroPolicy

# All engines are built on IdealCimMacroConfig, so the embedded macro policy is
# the empty marker. ``adc_bits == 0`` is the lossless sentinel: no ADC
# quantization, so engine outputs equal ``torch.matmul`` exactly.
_IDEAL_MACRO_POLICY = IdealCimMacroPolicy()
_ADC_MODE = 0
_ADC_BITS = 0


def _ideal_macro_config(
    *,
    col_num: int = 8,
    row_num: int = 8,
    active_row_num: int = 2,
) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        col_num=col_num,
        row_num=row_num,
        active_row_num=active_row_num,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        x_value_range=(0, 1),
        w_digit_count=1,
        w_digit_radix=4,
        w_digit_value_range=(-3, 3),
        adc_mode_num=1,
        adc_max_bits=0,
    )


def _zero_ppa() -> dict[str, float]:
    return {
        "energy_per_op__fJ": 0.0,
        "latency_per_op__ns": 0.0,
        "leakage_per_inst__uW": 0.0,
        "area_per_inst__um2": 0.0,
    }


def _accumulator_config() -> AccumulatorConfig:
    return AccumulatorConfig(bit_width=32, **_zero_ppa())


def _shift_adder_config() -> ShiftAdderConfig:
    return ShiftAdderConfig(bit_width=32, **_zero_ppa())


def _engine_kwargs(w_logical_shape: tuple[int, ...], policy: CimEnginePolicy) -> dict[str, Any]:
    return {
        "policy": policy,
        "w_logical_shape": w_logical_shape,
        "dtype": torch.float32,
        "T__K": 300.0,
        "ideal_macro": False,
    }


def _build_direct(
    *,
    w_logical_shape: tuple[int, ...],
    row_num: int = 8,
    active_row_num: int = 2,
) -> DirectCimEngine:
    config = DirectCimEngineConfig(
        cim_macro_config=_ideal_macro_config(row_num=row_num, active_row_num=active_row_num),
        w_encoding=Encoding.TRUE_FORM,
        col_accumulator_config=_accumulator_config(),
        phase_accumulator_config=_accumulator_config(),
    )
    engine = DirectCimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            DirectCimEnginePolicy(cim_macro_policy=_IDEAL_MACRO_POLICY),
        ),
    )
    engine.eval()
    return engine


def _slice_config_kwargs(*, row_num: int, active_row_num: int) -> dict[str, Any]:
    return {
        "cim_macro_config": _ideal_macro_config(row_num=row_num, active_row_num=active_row_num),
        "w_slice_num": 2,
        "x_slice_num": 2,
        "w_encoding": "true_form",
        "col_accumulator_config": _accumulator_config(),
        "phase_accumulator_config": _accumulator_config(),
        "sa_shift_adder_config": _shift_adder_config(),
        "sw_shift_adder_config": _shift_adder_config(),
    }


def _build_inter(
    *,
    w_logical_shape: tuple[int, ...],
    row_num: int = 8,
    active_row_num: int = 2,
) -> InterArraySliceCimEngine:
    config = InterArraySliceCimEngineConfig(**_slice_config_kwargs(row_num=row_num, active_row_num=active_row_num))
    engine = InterArraySliceCimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            InterArraySliceCimEnginePolicy(cim_macro_policy=_IDEAL_MACRO_POLICY),
        ),
    )
    engine.eval()
    return engine


def _build_intra(
    *,
    w_logical_shape: tuple[int, ...],
    row_num: int = 8,
    active_row_num: int = 2,
) -> IntraArraySliceCimEngine:
    config = IntraArraySliceCimEngineConfig(**_slice_config_kwargs(row_num=row_num, active_row_num=active_row_num))
    engine = IntraArraySliceCimEngine(
        config=config,
        **_engine_kwargs(
            w_logical_shape,
            IntraArraySliceCimEnginePolicy(cim_macro_policy=_IDEAL_MACRO_POLICY),
        ),
    )
    engine.eval()
    return engine


def _randint_in_range(value_range: tuple[int, int], shape: tuple[int, ...]) -> torch.Tensor:
    lo, hi = value_range
    return torch.randint(lo, hi + 1, shape, dtype=torch.int32)


def _assert_engine_matches_torch(engine: CimEngine, weight: torch.Tensor, activation: torch.Tensor) -> None:
    engine.program(weight)
    actual = engine.matmul(activation, adc_mode=_ADC_MODE, adc_bits=_ADC_BITS)
    expected = torch.matmul(activation.to(torch.int64), weight.transpose(-1, -2).to(torch.int64))
    assert actual.shape == expected.shape
    assert torch.equal(actual.to(torch.int64), expected)


def test_sub_phase_mask_content() -> None:
    """row_num=8, active_row_num=2: P=4 and mask[p, r] == (r // 2 == p)."""
    engine = _build_direct(w_logical_shape=(4, 8), row_num=8, active_row_num=2)
    assert engine._sub_phase_num == 4
    mask = engine._active_row_mask
    assert mask.shape == (4, 8)
    rows = torch.arange(8)
    phases = torch.arange(4).unsqueeze(-1)
    assert torch.equal(mask, rows // 2 == phases)


def test_unroll_sub_phase_layout() -> None:
    """P lands immediately left of the inst-aligned block; planes partition rows."""
    engine = _build_direct(w_logical_shape=(4, 8), row_num=8, active_row_num=2)
    p = engine._sub_phase_num
    # Organized activation layout for the direct engine: [M, Tc, Tr, row_num].
    x = torch.arange(1, 17, dtype=torch.int64).reshape(2, 1, 1, 8)
    planes = engine._unroll_sub_phase(x)
    assert planes.shape == (p, 2, 1, 1, 8)
    # Masks partition the row axis: summing P restores the full plane.
    assert torch.equal(planes.sum(dim=0), x)
    # Zero-fill outside each sub-phase's active window (WL off).
    mask = engine._active_row_mask
    for phase in range(p):
        assert torch.equal(planes[phase], torch.where(mask[phase], x, torch.zeros_like(x)))


def test_unroll_sub_phase_batch_axes_stay_left_of_p() -> None:
    """Caller batch axes ride left of P; P stays adjacent to the inst block."""
    engine = _build_direct(w_logical_shape=(4, 8), row_num=8, active_row_num=2)
    x = torch.randint(0, 2, (3, 2, 1, 1, 8), dtype=torch.int64)
    planes = engine._unroll_sub_phase(x)
    assert planes.shape == (3, 4, 2, 1, 1, 8)
    assert torch.equal(planes.sum(dim=1), x)


def test_sub_phase_non_divisible_ceil_covers_all_rows() -> None:
    """row_num=10, active_row_num=3: P = ceil(10/3) = 4 and every row lands in
    exactly one sub-phase (the last block is the short remainder)."""
    engine = _build_direct(w_logical_shape=(4, 10), row_num=10, active_row_num=3)
    assert engine._sub_phase_num == 4
    mask = engine._active_row_mask
    assert mask.shape == (4, 10)
    # Each of the 10 rows belongs to exactly one sub-phase — all rows covered.
    assert torch.equal(mask.sum(dim=0), torch.ones(10, dtype=mask.dtype))
    # The short final block owns only its single real row (row 9).
    assert int(mask[3].sum()) == 1


def test_degenerate_sub_phase_axis_size_one() -> None:
    """active_row_num == row_num: the P axis is still present with size 1."""
    engine = _build_direct(w_logical_shape=(4, 8), row_num=8, active_row_num=8)
    assert engine._sub_phase_num == 1
    x = torch.randint(0, 2, (2, 1, 1, 8), dtype=torch.int64)
    planes = engine._unroll_sub_phase(x)
    assert planes.shape == (1, 2, 1, 1, 8)
    assert torch.equal(planes[0], x)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_sub_phase_count_skips_padding_only_blocks(build: Callable[..., CimEngine]) -> None:
    """K < row_num runs only ceil(K / max_rows) sub-phases, not the full
    ceil(row_num / max_rows): the trailing blocks would read only zero-padded
    rows, so the engine skips them (efficiency + network-eval energy fidelity).

    row_num=8, active_row_num=2, K=3: min block count = ceil(3/2) = 2, well
    below the full ceil(8/2) = 4. Correctness is still checked against the
    torch oracle: the skipped blocks contributed exactly 0."""
    torch.manual_seed(3)
    n, k, m = 4, 3, 3  # k < row_num: a single short tile, most blocks empty
    engine = build(w_logical_shape=(n, k), row_num=8, active_row_num=2)
    assert engine._sub_phase_num == 2  # ceil(3/2), not ceil(8/2) == 4
    assert engine._active_row_mask.shape == (2, 8)
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_with_sub_phases(build: Callable[..., CimEngine]) -> None:
    """Lossless multi-sub-phase matmul equals the torch.matmul oracle."""
    torch.manual_seed(7)
    n, k, m = 5, 10, 3  # k > row_num exercises Tc tiling alongside P
    engine = build(w_logical_shape=(n, k), row_num=8, active_row_num=2)
    assert engine._sub_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("build", [_build_direct, _build_inter, _build_intra])
def test_engine_matmul_parity_non_divisible(build: Callable[..., CimEngine]) -> None:
    """Non-divisible geometry still equals the torch.matmul oracle: P via ceil
    covers every row, so k == row_num == 10 populated rows all contribute. The
    old floor division (P = 3) would drop row 9 and mismatch."""
    torch.manual_seed(9)
    n, k, m = 5, 10, 3  # k == row_num: all 10 rows carry real weight
    engine = build(w_logical_shape=(n, k), row_num=10, active_row_num=3)
    assert engine._sub_phase_num == 4
    weight = _randint_in_range(engine.w_value_range, (n, k))
    activation = _randint_in_range(engine.x_value_range, (m, k))
    _assert_engine_matches_torch(engine, weight, activation)


@pytest.mark.parametrize("activation_batch", [(), (2,)])
def test_direct_engine_weight_batch_parity(activation_batch: tuple[int, ...]) -> None:
    """w_batch=(2,) pins the -(b+5) sub-phase dim under weight-batch prefixes."""
    torch.manual_seed(11)
    b, n, k, m = 2, 5, 10, 3
    engine = _build_direct(w_logical_shape=(b, n, k), row_num=8, active_row_num=2)
    weight = _randint_in_range(engine.w_value_range, (b, n, k))
    activation = _randint_in_range(engine.x_value_range, (*activation_batch, m, k))
    _assert_engine_matches_torch(engine, weight, activation)

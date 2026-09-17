"""Which axis of the macro access schedule `(M, Sx, D, P)` multiplies which digital block's window.

In `Conv2dCimUnit.latency__ns`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neurox import Profiler, stamp_names
from neurox.architecture.mapping import (
    TilingMode,
)
from neurox.architecture.unit.conv2d import Conv2dCimUnit, Conv2dCimUnitConfig, Conv2dCimUnitPolicy
from neurox.encoding import Encoding
from neurox.primitive.digital import AccumulatorConfig, ShiftAdderConfig
from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)

# The ideal macro's own latency is zero, so every term the unit adds is a
# digital one and stays separable.
_ADC_BITS = None
_QUANTIZATION_MODE = 0

_INPUT_NUM = 8
_OUTPUT_NUM = 4

# Hand-derived placement of `(N, K) = (8, 4)` on an 8-input macro whose weight
# block spans all 4 output ports: L = min(4, 8) = 4, so Tc = 1; two output
# blocks fit one group of capacity 2, so G = 1 and D = 2.
_W_SHAPE = (8, 4)
_BLOCK_STEP_NUM = 2
# The intra-port layout halves the block width to Q = 2, giving four output
# blocks over two groups: G = 2, D = 2.
_INTRA_AGGREGATED_PORT_NUM = 2


def _macro_config(*, max_active_num: int) -> IdealCimMacroConfig:
    return IdealCimMacroConfig(
        input_num=_INPUT_NUM,
        rescale_factors=(1.0,),
        max_active_num=max_active_num,
        lane_num=1,
        scan_num=_OUTPUT_NUM,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
        w_digit_num=2,
        w_digit_radix=2,
        w_encoding=Encoding.TRUE_FORM,
        x_digit_num=1,
        x_digit_radix=2,
        x_encoding=Encoding.UNSIGNED,
        x_value_range=(0, 1),
        w_value_range=(-3, 3),
        adc_bits=8,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )


def _accumulator_config(*, latency_per_op__ns: float, energy_per_op__fJ: float = 0.0) -> AccumulatorConfig:
    return AccumulatorConfig(
        bit_width=32,
        energy_per_op__fJ=energy_per_op__fJ,
        latency_per_op__ns=latency_per_op__ns,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _shift_adder_config(*, latency_per_op__ns: float) -> ShiftAdderConfig:
    return ShiftAdderConfig(
        bit_width=32,
        energy_per_op__fJ=0.0,
        latency_per_op__ns=latency_per_op__ns,
        leakage_per_inst__uW=0.0,
        area_per_inst__um2=0.0,
    )


def _unit_config(
    *,
    w_slice_num: int,
    w_slice_encoding: Encoding | None,
    x_slice_num: int,
    x_slice_encoding: Encoding | None,
    tiling: TilingMode,
    w_shift_adder_config: ShiftAdderConfig | None,
    x_shift_adder_config: ShiftAdderConfig | None,
    max_active_num: int = 4,
    phase__ns: float = 0.0,
    phase_energy__fJ: float = 0.0,
) -> Conv2dCimUnitConfig:
    return Conv2dCimUnitConfig(
        cim_macro_config=_macro_config(max_active_num=max_active_num),
        merge=True,
        phase_accumulator_config=_accumulator_config(latency_per_op__ns=phase__ns, energy_per_op__fJ=phase_energy__fJ),
        w_slice_num=w_slice_num,
        w_slice_encoding=w_slice_encoding,
        x_slice_num=x_slice_num,
        x_slice_encoding=x_slice_encoding,
        tiling=tiling,
        w_shift_adder_config=w_shift_adder_config,
        x_shift_adder_config=x_shift_adder_config,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
    )


def _unit_policy() -> Conv2dCimUnitPolicy:
    return Conv2dCimUnitPolicy(
        cim_macro_policy=IdealCimMacroPolicy(),
    )


def _build_pointwise(unit_config: Conv2dCimUnitConfig, *, w_logical_shape: tuple[int, ...] = _W_SHAPE) -> Conv2dCimUnit:
    config = replace(unit_config, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0)
    unit = Conv2dCimUnit(
        config=config,
        policy=_unit_policy(),
        w_logical_shape=(*w_logical_shape, 1, 1),
        dtype=torch.float32,
    )
    unit.eval()
    return unit


def _build_conv2d(unit_config: Conv2dCimUnitConfig, *, w_logical_shape: tuple[int, int, int, int]) -> Conv2dCimUnit:
    config = replace(
        unit_config, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
    )
    unit = Conv2dCimUnit(
        config=config,
        policy=_unit_policy(),
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
    )
    unit.eval()
    return unit


@pytest.mark.parametrize(("merge", "expected__ns"), [(False, 12.0), (True, 15.0)])
def test_partial_output_tiles_retire_at_their_actual_scan_count(monkeypatch, merge, expected__ns):
    config = _unit_config(
        w_slice_num=1,
        w_slice_encoding=None,
        w_shift_adder_config=None,
        tiling=TilingMode.SLICE_PLANES,
        x_slice_num=1,
        x_slice_encoding=None,
        x_shift_adder_config=None,
    )
    config = replace(config, merge=merge)
    unit = _build_pointwise(config, w_logical_shape=(9, 4))
    monkeypatch.setattr(unit.cim_macro, "_latency_per_scan__ns", lambda **kwargs: 3.0)
    # Nine outputs occupy three blocks of four, four, and one. Without
    # sharing they finish in parallel after four scans. Sharing adds one
    # serial access for the final output after the two full blocks finish.
    assert unit.latency__ns((4, 1, 1), adc_active_bits=None) == expected__ns


# --- The schedule the unit multiplies ---


def test_direct_layout_times_the_phase_accumulator() -> None:
    """D and the port count are the only multipliers a direct/direct layout has, neither slice stage owning a block."""
    phase__ns = 0.25
    unit = _build_pointwise(
        _unit_config(
            w_slice_num=1,
            w_slice_encoding=None,
            w_shift_adder_config=None,
            tiling=TilingMode.SLICE_PLANES,
            x_slice_num=1,
            x_slice_encoding=None,
            x_shift_adder_config=None,
            max_active_num=4,
            phase__ns=phase__ns,
        )
    )
    assert unit.merge.merge_step_num == _BLOCK_STEP_NUM
    assert unit.input_activation.input_phase_num == 1
    assert unit.x_slicer.slice_num == 1

    expected__ns = _BLOCK_STEP_NUM * _OUTPUT_NUM * phase__ns
    assert unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


def test_phase_accumulator_runs_once_per_arrival() -> None:
    """P multiplies the serial register's window and nothing else's."""

    def _phase_only(max_active_num: int) -> float:
        unit = _build_pointwise(
            _unit_config(
                w_slice_num=1,
                w_slice_encoding=None,
                w_shift_adder_config=None,
                tiling=TilingMode.SLICE_PLANES,
                x_slice_num=1,
                x_slice_encoding=None,
                x_shift_adder_config=None,
                max_active_num=max_active_num,
                phase__ns=1.0,
            )
        )
        return unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS)

    assert _phase_only(2) == pytest.approx(2.0 * _phase_only(4))


# --- The recombination blocks ---


def test_inter_layout_recombines_once_per_step_it_closes() -> None:
    """Both reconstructions are one-pass positional sums.

    The weight-slice block runs once per input slice, the input-slice block once per block step.
    """
    x_slice_num = 3
    phase__ns = 0.25
    w_recombine__ns = 1.0
    x_recombine__ns = 2.0
    unit = _build_pointwise(
        _unit_config(
            w_slice_num=2,
            w_slice_encoding=Encoding.TRUE_FORM,
            w_shift_adder_config=_shift_adder_config(latency_per_op__ns=w_recombine__ns),
            tiling=TilingMode.SLICE_PLANES,
            x_slice_num=x_slice_num,
            x_slice_encoding=Encoding.UNSIGNED,
            x_shift_adder_config=_shift_adder_config(latency_per_op__ns=x_recombine__ns),
            max_active_num=4,
            phase__ns=phase__ns,
        )
    )
    assert unit.x_slicer.slice_num == x_slice_num
    assert unit.merge.merge_step_num == _BLOCK_STEP_NUM
    assert unit.tiling.logical_tile_output_num == _OUTPUT_NUM

    step_num = x_slice_num * _BLOCK_STEP_NUM
    expected__ns = step_num * _OUTPUT_NUM * (phase__ns + w_recombine__ns)
    expected__ns += _BLOCK_STEP_NUM * _OUTPUT_NUM * x_recombine__ns
    assert unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


def test_input_slice_recombination_ignores_the_slice_count() -> None:
    """Sx is the digit axis of the one pass that removes it, so it never multiplies."""

    def _x_recombine_only(x_slice_num: int) -> float:
        unit = _build_pointwise(
            _unit_config(
                w_slice_num=1,
                w_slice_encoding=None,
                w_shift_adder_config=None,
                tiling=TilingMode.SLICE_PLANES,
                x_slice_num=x_slice_num,
                x_slice_encoding=Encoding.UNSIGNED,
                x_shift_adder_config=_shift_adder_config(latency_per_op__ns=1.0),
                max_active_num=4,
            )
        )
        return unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS)

    assert _x_recombine_only(5) == pytest.approx(_x_recombine_only(2))
    assert _x_recombine_only(2) == pytest.approx(_BLOCK_STEP_NUM * _OUTPUT_NUM)


def test_intra_layout_recombines_over_the_ports_one_aggregation_leaves() -> None:
    """An intra-port layout leaves `output_num // w_slice_num` logical outputs.

    The reconstruction runs over those, not the macro ports.
    """
    unit = _build_pointwise(
        _unit_config(
            w_slice_num=2,
            w_slice_encoding=Encoding.TRUE_FORM,
            w_shift_adder_config=_shift_adder_config(latency_per_op__ns=1.0),
            tiling=TilingMode.SLICE_OUTPUTS,
            x_slice_num=1,
            x_slice_encoding=None,
            x_shift_adder_config=None,
            max_active_num=4,
        )
    )
    assert unit.tiling.logical_tile_output_num == _INTRA_AGGREGATED_PORT_NUM
    assert unit.merge.merge_step_num == _BLOCK_STEP_NUM

    expected__ns = _BLOCK_STEP_NUM * _INTRA_AGGREGATED_PORT_NUM
    assert unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS) == pytest.approx(expected__ns)


# --- The one runtime extent ---


def test_output_planes_multiply_the_whole_schedule() -> None:
    """M = H_out · W_out enters at the conv unit and multiplies every unit term, while a caller batch does not."""
    unit_config = _unit_config(
        w_slice_num=1,
        w_slice_encoding=None,
        w_shift_adder_config=None,
        tiling=TilingMode.SLICE_PLANES,
        x_slice_num=1,
        x_slice_encoding=None,
        x_shift_adder_config=None,
        max_active_num=4,
        phase__ns=0.25,
    )
    # (C_out, C_in, kh, kw) contracting to K = 4, the linear fixtures' shape.
    unit = _build_conv2d(unit_config, w_logical_shape=(8, 1, 2, 2))
    one_plane__ns = unit.latency__ns((1, 2, 2), adc_active_bits=_ADC_BITS)

    assert unit.latency__ns((1, 4, 4), adc_active_bits=_ADC_BITS) == pytest.approx(9 * one_plane__ns)
    assert unit.latency__ns((1, 5, 4), adc_active_bits=_ADC_BITS) == pytest.approx(12 * one_plane__ns)
    assert unit.latency__ns((7, 1, 4, 4), adc_active_bits=_ADC_BITS) == pytest.approx(9 * one_plane__ns)


# --- The round count against a measured forward ---


def test_phase_accumulator_rounds_match_the_measured_forward(device: torch.device) -> None:
    """The round count multiplying latency equals the operands a measured forward folds.

    Per instance and per caller operation.
    """
    phase__ns = 0.75
    phase_energy__fJ = 3.0
    unit = _build_pointwise(
        _unit_config(
            w_slice_num=2,
            w_slice_encoding=Encoding.TRUE_FORM,
            w_shift_adder_config=_shift_adder_config(latency_per_op__ns=0.0),
            tiling=TilingMode.SLICE_PLANES,
            x_slice_num=3,
            x_slice_encoding=Encoding.UNSIGNED,
            x_shift_adder_config=_shift_adder_config(latency_per_op__ns=0.0),
            max_active_num=2,
            phase__ns=phase__ns,
            phase_energy__fJ=phase_energy__fJ,
        )
    )
    accumulator = unit.phase_accumulator
    batch_num = 5
    torch.manual_seed(0)
    weight = torch.randint(-3, 4, _W_SHAPE, dtype=torch.int32)
    x = torch.randint(0, 2, (batch_num, _W_SHAPE[1]), dtype=torch.int32)
    unit.to(device)
    unit.program(weight.to(device)[..., None, None])
    stamp_names(unit)
    accumulator_name = accumulator.qualified_name

    with Profiler(leading_rank=1) as profiler:
        unit.conv2d(x.to(device)[..., None, None], quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    folded_operand_num = (
        sum(
            float(record.dynamic_energy__fJ.sum())
            for record in profiler.records
            if record.qualified_name == accumulator_name
        )
        / phase_energy__fJ
    )
    round_num = folded_operand_num / (batch_num * accumulator.inst_count)

    latency__ns = float(unit.latency__ns((_W_SHAPE[1], 1, 1), adc_active_bits=_ADC_BITS))
    assert latency__ns == pytest.approx(round_num * phase__ns)


def test_unit_owned_shift_adders_preserve_wrap_and_energy(device: torch.device) -> None:
    unit = _build_pointwise(
        _unit_config(
            w_slice_num=2,
            w_slice_encoding=Encoding.TRUE_FORM,
            x_slice_num=2,
            x_slice_encoding=Encoding.UNSIGNED,
            tiling=TilingMode.SLICE_PLANES,
            w_shift_adder_config=replace(
                _shift_adder_config(latency_per_op__ns=0.0), bit_width=4, energy_per_op__fJ=2.0
            ),
            x_shift_adder_config=replace(
                _shift_adder_config(latency_per_op__ns=0.0), bit_width=4, energy_per_op__fJ=3.0
            ),
        ),
        w_logical_shape=(1, 1),
    ).to(device)
    unit.program(torch.tensor([[[[7]]]], dtype=torch.int32, device=device))
    stamp_names(unit)
    x = torch.tensor([[3], [3]], dtype=torch.int32, device=device)
    with Profiler(leading_rank=1) as profiler:
        actual = unit.conv2d(x[..., None, None], quantization_mode=_QUANTIZATION_MODE, adc_active_bits=_ADC_BITS)
    # 7 * 3 = 21 wraps to 5 in the final four-bit register.
    torch.testing.assert_close(actual, torch.full_like(actual, 5))
    costs = {
        adder.qualified_name: sum(
            float(record.dynamic_energy__fJ.sum())
            for record in profiler.records
            if record.qualified_name == adder.qualified_name
        )
        for adder in (unit.w_shift_adder, unit.x_shift_adder)
    }
    # Two vectors, two x slices and two w slices feed the w reconstruction;
    # the x reconstruction consumes two slices of each final output.
    assert costs[unit.w_shift_adder.qualified_name] == 16.0
    assert costs[unit.x_shift_adder.qualified_name] == 12.0

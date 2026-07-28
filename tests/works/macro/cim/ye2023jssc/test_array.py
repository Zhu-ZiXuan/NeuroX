"""CPU-only eager tests for the ye2023jssc dedicated WH-2T1R array.

Laws (config = arbitrary hand-written witness, not the assertion target):

  * ``solve`` produces one summed T2 compute current per output row; each
    ``i_tbl[o]`` equals the place-value-weighted lookup SUM over the active
    row's cells, computed independently from the witness table — with the
    REDUNDANT plane's place values appended after the weight planes, so the
    column place-value vector spans ``weight_radix + redundant_radix``,
  * place value is analog: flipping a cell in the m = 2 plane shifts ``i_tbl`` by
    ~2x the shift of the same flip in the m = 1 plane, and the redundant plane
    carries its own configured place value,
  * step2 gates on the CONFIGURED WL threshold: a drive below it selects no row
    and ``i_tbl`` collapses to zero,
  * the returned steady state carries ``i_bl_port__uA`` / ``v_bl_clamp__V`` /
    ``i_tbl__uA`` and NO ``v_x`` (chunk-fusion keeps V_X per chunk),
  * under a profiler the array bills its PER-ACCESS capacitance as a single
    un-channelled event — the WL wire segments of the selected row plus the cell
    terms, and nothing else: no conduction (invariant to any window), no SL
    trapezoid, no BL wire / BL node charge (invariant to ``c_sl__fF``,
    ``c_bl__fF`` and the BL wire caps). At zero input it collapses to the exact
    closed-form WL-only value.

Runs eagerly (dynamo disabled) so the compiled solver leaf is not unrolled; tiny
CPU shapes, float64.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox.common.profiler import NeuroxProfiler
from neurox.primitive.analog.voltage_driver import (
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
)
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig
from neurox.works.macro.cim.ye2023jssc.array import (
    Ye2023Jssc2t1rArray,
    Ye2023Jssc2t1rArrayConfig,
    Ye2023Jssc2t1rArrayPolicy,
    Ye2023Jssc2t1rSteadyState,
)
from neurox.works.macro.cim.ye2023jssc.cell import (
    Ye2023Jssc2t1rCellConfig,
    Ye2023Jssc2t1rCellPolicy,
)

_DTYPE = torch.float64

# --- Tiny witness geometry ---
_INPUT_NUM = 2
_WEIGHT_RADIX = (1, 2, 3)  # LSB-first weight place values
_REDUNDANT_RADIX = (5,)  # the non-weight plane, appended last
_ALL_RADIX = (*_WEIGHT_RADIX, *_REDUNDANT_RADIX)
_COL_NUM = _INPUT_NUM * len(_ALL_RADIX)  # 8 physical columns
_ROW_NUM = 4  # 4 output rows

# --- WH-2T1R cell witness currents (arbitrary; laws asserted, not these) ---
_HRS = 0
_LRS = 1
_FLOOR__uA = 0.01  # IN=0 off-cell floor, state-independent
_LEAK_IN1__uA = 0.02  # (HRS, IN=1) weight leakage
_UNIT_IN1__uA = 0.5  # (LRS, IN=1) unit-scale on-current

_V_WL_SEL__V = 0.6
_V_WL_ON_THRESHOLD__V = 0.3
_V_BL_IN1__V = 0.3
_V_BL_THRESHOLD__V = 0.15

_C_WL__fF = 1.0
_C_X__fF = 1.0
_C_BL__fF = 1.0
_C_SL__fF = 1.0
_WL_FIRST_C__fF = 0.4
_WL_SEGMENT_C__fF = 0.1
_BL_FIRST_C__fF = 0.4
_BL_SEGMENT_C__fF = 0.1

# Tiny positive wire R (solver needs R > 0); small so the chain stays near-ideal.
_WIRE_FIRST_R__MOhm = 2.0e-5
_WIRE_SEGMENT_R__MOhm = 5.0e-6


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _cell_config() -> Ye2023Jssc2t1rCellConfig:
    return Ye2023Jssc2t1rCellConfig(
        c_bl__fF=_C_BL__fF,
        c_x__fF=_C_X__fF,
        c_sl__fF=_C_SL__fF,
        c_wl__fF=_C_WL__fF,
        g_cell_off_table__uS=(1.0, 5.0),
        g_cell_on_table__uS=(2.0, 10.0),
        vx_ratio_off_table=(0.3, 0.4),
        vx_ratio_on_table=(0.5, 0.6),
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
        # i_t2_table__uA[input_bit][state]; state axis = (HRS, LRS).
        i_t2_table__uA=(
            (_FLOOR__uA, _FLOOR__uA),  # IN=0: state-independent floor
            (_LEAK_IN1__uA, _UNIT_IN1__uA),  # IN=1: HRS leak vs LRS unit current
        ),
    )


def _array_config() -> Ye2023Jssc2t1rArrayConfig:
    return Ye2023Jssc2t1rArrayConfig(
        row_first_space__um=1.0,
        row_cell_space__um=1.0,
        col_first_space__um=1.0,
        col_cell_space__um=1.0,
        bl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        bl_first_c__fF=_BL_FIRST_C__fF,
        bl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        bl_segment_c__fF=_BL_SEGMENT_C__fF,
        sl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        sl_first_c__fF=0.4,
        sl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        sl_segment_c__fF=0.1,
        wl_first_r__MOhm=_WIRE_FIRST_R__MOhm,
        wl_first_c__fF=_WL_FIRST_C__fF,
        wl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        wl_segment_c__fF=_WL_SEGMENT_C__fF,
        cell_config=_cell_config(),
        solver_config=NestedParallelRailSolverConfig(n_outer=2, n_inner=1),
        latency_per_op__ns=1.0,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=1.0,
        weight_radix=_WEIGHT_RADIX,
        redundant_radix=_REDUNDANT_RADIX,
        v_bl_in1__V=_V_BL_IN1__V,
        v_bl_in_threshold__V=_V_BL_THRESHOLD__V,
    )


def _ideal_clamp() -> VoltageDriver:
    clamp = VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(),
        dtype=_DTYPE,
        T__K=300.0,
    )
    clamp.eval()
    clamp.fabricate()
    return clamp


def _build_array(
    config: Ye2023Jssc2t1rArrayConfig | None = None,
    *,
    enable_latency_record: bool = True,
) -> Ye2023Jssc2t1rArray:
    array = Ye2023Jssc2t1rArray(
        config=config if config is not None else _array_config(),
        policy=Ye2023Jssc2t1rArrayPolicy(cell_policy=Ye2023Jssc2t1rCellPolicy(), solve_chunk_size=0),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        dtype=_DTYPE,
        T__K=300.0,
        enable_latency_record=enable_latency_record,
    )
    array.eval()
    array.fabricate()
    return array


def _one_hot_wl(v_wl_sel__V: float = _V_WL_SEL__V) -> torch.Tensor:
    """One-hot WL per output stacked onto the leading: ``v_wl_sel * eye(row_num)``."""
    return v_wl_sel__V * torch.eye(_ROW_NUM, dtype=_DTYPE)


def _bl_v_ref(input_bits: tuple[int, ...]) -> torch.Tensor:
    """Per-column BL input voltages, tiled plane-major over every plane."""
    per_input = torch.tensor([_V_BL_IN1__V if b else 0.0 for b in input_bits], dtype=_DTYPE)
    return per_input.repeat(len(_ALL_RADIX))  # [in0,in1, in0,in1, ...]


def _solve(array: Ye2023Jssc2t1rArray, v_wl: torch.Tensor, bl_v_ref: torch.Tensor) -> Ye2023Jssc2t1rSteadyState:
    clamp = _ideal_clamp()
    steady = array.solve(
        v_wl,
        bl_driver=clamp,
        bl_v_ref__V=bl_v_ref,
        sl_driver=clamp,
        sl_v_ref__V=torch.tensor(0.0, dtype=_DTYPE),
    )
    assert isinstance(steady, Ye2023Jssc2t1rSteadyState)
    return steady


def _expected_i_tbl(state: torch.Tensor, input_bits: tuple[int, ...]) -> torch.Tensor:
    """Independent oracle: per output o, sum_col radix[col] * lookup(input_high[col], state[col,o])."""
    table = torch.tensor(
        ((_FLOOR__uA, _FLOOR__uA), (_LEAK_IN1__uA, _UNIT_IN1__uA)),
        dtype=_DTYPE,
    )
    radix = torch.tensor(_ALL_RADIX, dtype=_DTYPE).repeat_interleave(_INPUT_NUM)  # [1,1,2,2,3,3,5,5]
    input_high = _bl_v_ref(input_bits) > _V_BL_THRESHOLD__V  # [col]
    i_t2_in1 = table[1][state]  # [col, row]
    i_t2_in0 = table[0][state]  # [col, row]
    sel = torch.where(input_high.unsqueeze(-1), i_t2_in1, i_t2_in0)  # [col, row]
    scaled = sel * radix.unsqueeze(-1)  # [col, row]
    return scaled.sum(dim=0)  # [row] = per output


def test_i_tbl_matches_lookup_sum_law() -> None:
    """Core MAC + floor law: i_tbl[o] == the independent place-value lookup sum."""
    array = _build_array()
    # Distinct HRS/LRS pattern over (col, row).
    state = torch.tensor(
        [
            [_HRS, _LRS, _HRS, _LRS],
            [_LRS, _HRS, _LRS, _HRS],
            [_HRS, _HRS, _LRS, _LRS],
            [_LRS, _LRS, _HRS, _HRS],
            [_HRS, _LRS, _LRS, _HRS],
            [_LRS, _HRS, _HRS, _LRS],
            [_HRS, _HRS, _HRS, _HRS],
            [_LRS, _LRS, _LRS, _LRS],
        ],
        dtype=torch.long,
    )
    array.program(state)
    input_bits = (1, 0)  # input 0 high, input 1 low
    steady = _solve(array, _one_hot_wl(), _bl_v_ref(input_bits))

    assert tuple(steady.i_tbl__uA.shape) == (_ROW_NUM,)
    expected = _expected_i_tbl(state, input_bits)
    assert torch.allclose(steady.i_tbl__uA, expected)


def test_place_value_is_analog_over_weight_and_redundant_planes() -> None:
    """A cell flip shifts i_tbl by its plane's configured place value, redundant plane included."""
    input_bits = (1, 0)  # input 0 (the even columns) high
    v_wl = _one_hot_wl()
    bl = _bl_v_ref(input_bits)
    out_row = 1

    base_state = torch.zeros((_COL_NUM, _ROW_NUM), dtype=torch.long)  # all HRS
    array = _build_array()
    array.program(base_state)
    base = _solve(array, v_wl, bl).i_tbl__uA.clone()

    def flip_delta(plane: int) -> torch.Tensor:
        state = base_state.clone()
        state[plane * _INPUT_NUM, out_row] = _LRS  # the input-high column of that plane
        array.program(state)
        return (_solve(array, v_wl, bl).i_tbl__uA - base)[out_row]

    deltas = [flip_delta(plane) for plane in range(len(_ALL_RADIX))]
    assert deltas[0] > 0
    for plane, m in enumerate(_ALL_RADIX):
        assert torch.isclose(deltas[plane], m * deltas[0] / _ALL_RADIX[0]), (
            f"plane {plane} place value is not {m}: {deltas}"
        )


def test_input_bit_flip_raises_current() -> None:
    """Flipping an input bit low -> high raises i_tbl (floor -> on-current)."""
    v_wl = _one_hot_wl()
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)  # all LRS

    array = _build_array()
    array.program(state)
    low = _solve(array, v_wl, _bl_v_ref((0, 0))).i_tbl__uA  # both inputs low
    high = _solve(array, v_wl, _bl_v_ref((1, 0))).i_tbl__uA  # input 0 high

    assert torch.all(high > low)


def test_step2_gates_on_configured_wl_threshold() -> None:
    """A WL drive at or below the configured threshold selects no row: i_tbl == 0."""
    array = _build_array()
    array.program(torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long))
    below = _solve(array, _one_hot_wl(_V_WL_ON_THRESHOLD__V), _bl_v_ref((1, 1))).i_tbl__uA
    assert torch.equal(below, torch.zeros_like(below))
    above = _solve(array, _one_hot_wl(), _bl_v_ref((1, 1))).i_tbl__uA
    assert torch.all(above > 0.0)


def test_steady_state_fields() -> None:
    """The returned dataclass carries the three named fields and NO v_x."""
    array = _build_array()
    array.program(torch.zeros((_COL_NUM, _ROW_NUM), dtype=torch.long))
    steady = _solve(array, _one_hot_wl(), _bl_v_ref((1, 0)))

    assert tuple(steady.i_bl_port__uA.shape) == (_ROW_NUM, _COL_NUM)
    assert tuple(steady.v_bl_clamp__V.shape) == (_ROW_NUM, _COL_NUM)
    assert tuple(steady.i_tbl__uA.shape) == (_ROW_NUM,)
    assert not hasattr(steady, "v_x")
    assert not hasattr(steady, "v_x__V")


# ---------------------------------------------------------------------------
# Per-access capacitance billing
# ---------------------------------------------------------------------------


def _profiled_energy(
    config: Ye2023Jssc2t1rArrayConfig,
    *,
    input_bits: tuple[int, ...],
    state: torch.Tensor,
) -> float:
    array = _build_array(config)
    array.program(state)
    with NeuroxProfiler() as prof, torch.no_grad():
        _solve(array, _one_hot_wl(), _bl_v_ref(input_bits))
    return prof.report(array).energy_by_name.get("", 0.0)


def test_profiler_bills_per_access_caps_as_one_event() -> None:
    """The array bills nonzero capacitive energy as one un-channelled event; no conduction."""
    array = _build_array()
    array.program(torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long))

    with NeuroxProfiler() as prof, torch.no_grad():
        _solve(array, _one_hot_wl(), _bl_v_ref((1, 0)))
    report = prof.report(array)

    # The array is the profiled root -> named "".
    assert report.energy_by_name.get("", 0.0) > 0.0

    # Exactly one energy event from the array, un-channelled (caps only).
    array_events = [e for e in report.energy_events if e.module is array]
    assert len(array_events) == 1
    assert array_events[0].channel is None
    assert array_events[0].dynamic_energy__fJ > 0.0


def test_zero_input_caps_are_the_closed_form_wl_terms() -> None:
    """With every column input-low the array bills exactly the per-row WL wire + gate load."""
    config = _array_config()
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)
    got = _profiled_energy(config, input_bits=(0, 0), state=state)

    c_wl_wire__fF = _WL_FIRST_C__fF + (_COL_NUM - 1) * _WL_SEGMENT_C__fF
    per_access__fJ = (c_wl_wire__fF + _COL_NUM * _C_WL__fF) * _V_WL_SEL__V**2
    assert got == pytest.approx(_ROW_NUM * per_access__fJ)

    # Driving inputs high adds the selected row's X dip-recharge on top.
    assert _profiled_energy(config, input_bits=(1, 1), state=state) > got


def test_array_caps_ignore_sl_and_bl_capacitance() -> None:
    """No SL trapezoid and no BL charge in the array: only the WL side and the cell terms."""
    base = _array_config()
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)
    got = _profiled_energy(base, input_bits=(1, 0), state=state)

    heavy_sl_cell = dataclasses.replace(_cell_config(), c_sl__fF=10.0 * _C_SL__fF)
    heavy_bl_cell = dataclasses.replace(_cell_config(), c_bl__fF=10.0 * _C_BL__fF)
    variants = {
        "cell c_sl__fF": dataclasses.replace(base, cell_config=heavy_sl_cell),
        "cell c_bl__fF": dataclasses.replace(base, cell_config=heavy_bl_cell),
        "wire sl caps": dataclasses.replace(base, sl_first_c__fF=4.0, sl_segment_c__fF=1.0),
        "wire bl caps": dataclasses.replace(base, bl_first_c__fF=4.0, bl_segment_c__fF=1.0),
    }
    for label, config in variants.items():
        moved = _profiled_energy(config, input_bits=(1, 0), state=state)
        assert moved == pytest.approx(got), f"array caps moved with {label}"

    # The WL side and the cell X dip DO move it (the terms the array actually owns).
    wl_heavy = dataclasses.replace(base, wl_first_c__fF=4.0, wl_segment_c__fF=1.0)
    assert _profiled_energy(wl_heavy, input_bits=(1, 0), state=state) > got
    x_heavy = dataclasses.replace(base, cell_config=dataclasses.replace(_cell_config(), c_x__fF=10.0 * _C_X__fF))
    assert _profiled_energy(x_heavy, input_bits=(1, 0), state=state) > got

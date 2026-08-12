"""CPU-only eager tests for the ye2023jssc dedicated WH-2T1R array.

The array is the kernel 1T1R array plus ONE extension — the transpose-bitline
lookup sum — so these laws split the same way: the lookup and the steady-state
shape are the scheme's, the capacitive billing is the kernel's
``BL_IN_WL_SCAN`` law and is only checked here for the two facts this scheme
fixes (the mode, and the hold that the mode amortizes over).

Laws (config = arbitrary hand-written witness, not the assertion target):

  * ``solve_array`` produces one summed T2 compute current per output row; each
    ``i_tbl[o]`` equals the place-value-weighted lookup SUM over the active
    row's cells, computed independently from the witness table — with the
    REDUNDANT plane's place values appended after the weight planes, so the
    column place-value vector spans ``weight_radix + redundant_radix``,
  * place value is analog: flipping a cell in the m = 2 plane shifts ``i_tbl`` by
    ~2x the shift of the same flip in the m = 1 plane, and the redundant plane
    carries its own configured place value,
  * step2 gates on the CONFIGURED WL threshold read off the per-cell gate drive:
    a drive below it selects no row and ``i_tbl`` collapses to zero,
  * a driven cell's operating point follows its solved ``V_X``: a column held at
    0 V sits on the floor entry, a column driven above it on the drive entry,
  * the returned steady state extends the kernel one with ``i_tbl__uA`` and
    carries NO ``v_x`` (the internal node feeds nothing downstream),
  * under a profiler the array bills its capacitance as a single un-channelled
    event following the held-BL scan law: with every column input-low there is
    no hold to establish and the bill collapses to the closed-form WL node
    terms, while a held input pattern is established exactly ONCE per full row
    scan — every per-node total (BL, X, SL, WL) moves the bill,
  * chunk size is a memory knob — the port state, the lookup sum and the billed
    energy are bit-identical across chunk sizes, the un-chunked solve included.

Runs eagerly (dynamo disabled) so the compiled solver leaf is not unrolled; tiny
CPU shapes, float64.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo

from neurox import Profiler, Reporter, stamp_names
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy
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
_FLOOR__uA = 0.01  # V_X = 0 off-cell floor, state-independent
_LEAK_DRIVE__uA = 0.02  # (HRS, drive point) weight leakage
_UNIT_DRIVE__uA = 0.5  # (LRS, drive point) unit-scale on-current

_V_WL_SEL__V = 0.6
_V_WL_ON_THRESHOLD__V = 0.3
_V_BL_IN1__V = 0.3

# --- Driver rails (separate variables; the WL driver is 1-bit, so its rail is
# its own ON level, which makes the WL terms hand-computable) ---
_V_DD_WL__V = _V_WL_SEL__V
_V_DD_BL__V = 0.8

# Per-node capacitance totals [fF]: each cell node's junction plus that node's
# share of the line it hangs on.
_WL_NODE_C__fF = 1.1
_X_NODE_C__fF = 1.0
_BL_NODE_C__fF = 1.1
_SL_NODE_C__fF = 1.1

# Tiny positive wire R (solver needs R > 0); small so the chain stays near-ideal.
_WIRE_SEGMENT_R__MOhm = 5.0e-6


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _cell_config() -> Ye2023Jssc2t1rCellConfig:
    return Ye2023Jssc2t1rCellConfig(
        g_cell_off_table__uS=(1.0, 5.0),
        g_cell_on_table__uS=(2.0, 10.0),
        vx_ratio_off_table=(0.3, 0.4),
        vx_ratio_on_table=(0.5, 0.6),
        v_wl_on_threshold__V=_V_WL_ON_THRESHOLD__V,
        # i_t2_table__uA[operating_point][state]; state axis = (HRS, LRS).
        i_t2_table__uA=(
            (_FLOOR__uA, _FLOOR__uA),  # V_X = 0: state-independent floor
            (_LEAK_DRIVE__uA, _UNIT_DRIVE__uA),  # V_X > 0: HRS leak vs LRS unit current
        ),
    )


def _array_config() -> Ye2023Jssc2t1rArrayConfig:
    return Ye2023Jssc2t1rArrayConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        sl_segment_r__MOhm=_WIRE_SEGMENT_R__MOhm,
        bl_node_c__fF=_BL_NODE_C__fF,
        x_node_c__fF=_X_NODE_C__fF,
        sl_node_c__fF=_SL_NODE_C__fF,
        wl_node_c__fF=_WL_NODE_C__fF,
        cell_config=_cell_config(),
        solver_config=NestedParallelRailSolverConfig(n_outer=2, n_inner=1),
        weight_radix=_WEIGHT_RADIX,
        redundant_radix=_REDUNDANT_RADIX,
        v_bl_in1__V=_V_BL_IN1__V,
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


def _build_array(config: Ye2023Jssc2t1rArrayConfig | None = None, *, chunk_size: int = 0) -> Ye2023Jssc2t1rArray:
    array = Ye2023Jssc2t1rArray(
        config=config if config is not None else _array_config(),
        policy=Ye2023Jssc2t1rArrayPolicy(cell_policy=Ye2023Jssc2t1rCellPolicy(), solve_chunk_size=chunk_size),
        inst_shape=(),
        row_num=_ROW_NUM,
        col_num=_COL_NUM,
        v_dd_wl__V=_V_DD_WL__V,
        v_dd_bl__V=_V_DD_BL__V,
        dtype=_DTYPE,
        T__K=300.0,
    )
    array.eval()
    array.fabricate()
    stamp_names(array)  # the standalone array is its own root, named ""
    return array


def _one_hot_wl(v_wl_sel__V: float = _V_WL_SEL__V) -> torch.Tensor:
    """One-hot WL per output stacked onto the leading: ``v_wl_sel * eye(row_num)``.

    The stack IS one full row scan — ``row_num`` accesses under one held BL
    pattern — which is the contract the mode's amortization rests on.
    """
    return v_wl_sel__V * torch.eye(_ROW_NUM, dtype=_DTYPE)


def _bl_v_ref(input_bits: tuple[int, ...]) -> torch.Tensor:
    """Per-column BL input voltages, tiled plane-major over every plane."""
    per_input = torch.tensor([_V_BL_IN1__V if b else 0.0 for b in input_bits], dtype=_DTYPE)
    return per_input.repeat(len(_ALL_RADIX))  # [in0,in1, in0,in1, ...]


def _solve(array: Ye2023Jssc2t1rArray, v_wl: torch.Tensor, bl_v_ref: torch.Tensor) -> Ye2023Jssc2t1rSteadyState:
    """Snapshot both boundary clamps at the call's event shape and solve.

    The caller owns the event structure, so the snaps are taken here and the
    word-line drive is handed over as the full cell grid (one value per gate,
    a stride-0 expand of the per-row drive across the columns).
    """
    clamp = _ideal_clamp()
    leading = tuple(torch.broadcast_shapes(v_wl.shape[:-1], bl_v_ref.shape[:-1]))
    event_shape = (*leading, _COL_NUM)
    # Shape: [..., row] -> [..., col, row]
    v_wl_grid = v_wl.unsqueeze(-2).expand(*leading, _COL_NUM, _ROW_NUM)
    bl_snap = clamp.snapshot(v_ref__V=bl_v_ref.expand(event_shape), shape=event_shape)
    sl_snap = clamp.snapshot(v_ref__V=torch.zeros((), dtype=_DTYPE).expand(event_shape), shape=event_shape)
    steady = array.solve_array(
        v_wl_grid,
        bl_driver=clamp,
        bl_driver_snap=bl_snap,
        sl_driver=clamp,
        sl_driver_snap=sl_snap,
    )
    assert isinstance(steady, Ye2023Jssc2t1rSteadyState)
    return steady


def _expected_i_tbl(state: torch.Tensor, input_bits: tuple[int, ...]) -> torch.Tensor:
    """Independent oracle: per output o, sum_col radix[col] * i_t2(point[col], state[col,o])."""
    table = torch.tensor(
        ((_FLOOR__uA, _FLOOR__uA), (_LEAK_DRIVE__uA, _UNIT_DRIVE__uA)),
        dtype=_DTYPE,
    )
    radix = torch.tensor(_ALL_RADIX, dtype=_DTYPE).repeat_interleave(_INPUT_NUM)  # [1,1,2,2,3,3,5,5]
    # A column driven above 0 V puts its selected cell on the drive point; a
    # column held at 0 V leaves it on the floor.
    # Shape: [col]
    driven = _bl_v_ref(input_bits) > 0.0
    # Shape: [col, row]
    i_t2_drive = table[1][state]
    # Shape: [col, row]
    i_t2_floor = table[0][state]
    # Shape: [col, row]
    sel = torch.where(driven.unsqueeze(-1), i_t2_drive, i_t2_floor)
    # Shape: [col, row]
    scaled = sel * radix.unsqueeze(-1)
    # One entry per output.
    # Shape: [col, row] -> [row]
    return scaled.sum(dim=0)


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
    """Driving a column's BL up moves its selected cells off the floor point onto the drive one."""
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
    """The returned dataclass extends the kernel state with the lookup sum, and NO v_x."""
    array = _build_array()
    array.program(torch.zeros((_COL_NUM, _ROW_NUM), dtype=torch.long))
    steady = _solve(array, _one_hot_wl(), _bl_v_ref((1, 0)))

    # Both boundaries are reported so the caller can bill each of its own clamps.
    for port in (steady.i_bl_port__uA, steady.v_bl_clamp__V, steady.i_sl_port__uA, steady.v_sl_drive__V):
        assert tuple(port.shape) == (_ROW_NUM, _COL_NUM)
    assert tuple(steady.i_tbl__uA.shape) == (_ROW_NUM,)
    assert not hasattr(steady, "v_x")
    assert not hasattr(steady, "v_x__V")


# ---------------------------------------------------------------------------
# Capacitive billing — the kernel's held-BL scan law
# ---------------------------------------------------------------------------


def _profiled_energy(
    config: Ye2023Jssc2t1rArrayConfig,
    *,
    input_bits: tuple[int, ...],
    state: torch.Tensor,
) -> float:
    """Bill one FULL row scan: ``row_num`` one-hot accesses under one held BL pattern."""
    array = _build_array(config)
    array.program(state)
    with Profiler() as prof, torch.no_grad():
        _solve(array, _one_hot_wl(), _bl_v_ref(input_bits))
    return Reporter(array).by_name(prof).get("", 0.0)


def test_profiler_bills_caps_as_one_record() -> None:
    """The array bills nonzero capacitive energy as one un-channelled record; no conduction."""
    array = _build_array()
    array.program(torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long))

    with Profiler() as prof, torch.no_grad():
        _solve(array, _one_hot_wl(), _bl_v_ref((1, 0)))

    # The array is the reported root -> named "".
    assert Reporter(array).by_name(prof).get("", 0.0) > 0.0

    # Exactly one energy record from the array, un-channelled (caps only).
    array_records = [r for r in prof.records if r.qualified_name == array.qualified_name]
    assert len(array_records) == 1
    assert array_records[0].channel is None
    assert array_records[0].dynamic_energy__fJ > 0.0


def test_zero_input_caps_are_the_closed_form_wl_terms() -> None:
    """With every column input-low there is no hold, so only the WL side is billed.

    Both boundaries rest at 0 V and settle to 0 V, so every conduction-path
    displacement and the whole precharge vanish and the supply-draw law leaves
    the WL node total of every cell, at ``V_DD_WL * C * |V_WL|`` per driven gate.
    """
    config = _array_config()
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)
    got = _profiled_energy(config, input_bits=(0, 0), state=state)

    per_access__fJ = _COL_NUM * _WL_NODE_C__fF * _V_DD_WL__V * _V_WL_SEL__V
    assert got == pytest.approx(_ROW_NUM * per_access__fJ)

    # Holding an input pattern adds the conduction-path terms on top.
    assert _profiled_energy(config, input_bits=(1, 1), state=state) > got


def test_every_node_capacitance_moves_the_array_bill() -> None:
    """The array owns its whole node set: each per-node total moves its bill.

    The BL and X nodes move through the level the input is held at, the SL node
    through the IR-drop displacement off its grounded rest, the WL node through
    the gate drive.
    """
    base = _array_config()
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)
    got = _profiled_energy(base, input_bits=(1, 0), state=state)

    for field in ("bl_node_c__fF", "x_node_c__fF", "sl_node_c__fF", "wl_node_c__fF"):
        heavier = dataclasses.replace(base, **{field: 10.0})
        assert _profiled_energy(heavier, input_bits=(1, 0), state=state) > got, f"array caps blind to {field}"


def test_the_held_input_is_established_once_per_row_scan() -> None:
    """One hold covers one full row scan, so its establishment is billed ONCE, not per access.

    Each BL node is charged from ground to the held level exactly once per scan,
    so the scan's sensitivity to ``bl_node_c__fF`` is ONE array's worth of that
    charge — ``row_num`` times smaller than an unamortized per-access bill. The
    residual is the per-solve displacement off the held level, which is the wire
    IR drop alone.
    """
    base = _array_config()
    delta__fF = 3.0
    heavy = dataclasses.replace(base, bl_node_c__fF=_BL_NODE_C__fF + delta__fF)
    state = torch.ones((_COL_NUM, _ROW_NUM), dtype=torch.long)
    input_bits = (1, 1)  # every physical column held at V_BL_in1

    moved = _profiled_energy(heavy, input_bits=input_bits, state=state) - _profiled_energy(
        base, input_bits=input_bits, state=state
    )
    one_scan__fJ = _V_DD_BL__V * delta__fF * _COL_NUM * _ROW_NUM * _V_BL_IN1__V
    assert moved == pytest.approx(one_scan__fJ, rel=1e-2)
    # An unamortized hold would cost row_num of those.
    assert moved < 0.5 * _ROW_NUM * one_scan__fJ


# ---------------------------------------------------------------------------
# Chunked solving
# ---------------------------------------------------------------------------

_CHUNK_LEADING = 7  # coprime with every chunk size below, so tails are padded


def _chunk_drive() -> tuple[torch.Tensor, torch.Tensor]:
    """A leading of independent (one-hot WL, per-column BL input) pairs."""
    row_pick = torch.arange(_CHUNK_LEADING) % _ROW_NUM
    # Shape: [leading, row]
    v_wl = _one_hot_wl()[row_pick]
    # Shape: [leading, col]
    bl_v_ref = torch.stack([_bl_v_ref((i % 2, (i // 2) % 2)) for i in range(_CHUNK_LEADING)])
    return v_wl, bl_v_ref


def test_chunk_size_moves_neither_the_lookup_sum_nor_the_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk size is a memory knob: every chunking of one call agrees bit for bit."""
    v_wl, bl_v_ref = _chunk_drive()
    state = (torch.arange(_COL_NUM * _ROW_NUM) % 2).reshape(_COL_NUM, _ROW_NUM)
    folded: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for chunk_size in (0, 2, 3, 5, 100):
        array = _build_array(chunk_size=chunk_size)
        array.program(state)
        billed: list[torch.Tensor] = []
        monkeypatch.setattr(array, "_record_dynamic_energy", billed.append)
        with Profiler(), torch.no_grad():
            steady = _solve(array, v_wl, bl_v_ref)
        [energy__fJ] = billed
        folded[chunk_size] = (steady.i_bl_port__uA, steady.v_bl_clamp__V, steady.i_tbl__uA, energy__fJ)

    # Both foldings are billed once, at the full leading, after the fold.
    assert folded[0][2].shape == (_CHUNK_LEADING,)
    assert folded[0][3].shape == (_CHUNK_LEADING,)
    # `0` is the un-chunked solve: one measurement over the whole leading.
    whole = folded[0]
    for chunk_size, measured in folded.items():
        for field, (got, want) in enumerate(zip(measured, whole, strict=True)):
            assert torch.equal(got, want), (chunk_size, field)

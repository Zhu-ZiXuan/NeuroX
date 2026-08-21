"""Twin laws for the flattened xue2020jssc array: serialization commutes with one solve.

The column MUX is a MACRO axis. The array holds every physical column and
settles all of them in ONE solve per WL plane, while the macro keeps the
`[..., sweep, gn, polarity, w_digit]` seat layout its transcode, readout,
billing, and latency speak. That is only sound because the physical columns are
independent BL/SL ladders, so serializing them into `mux_factor` slots and
solving them together must produce the SAME numbers — which is what this file
executes rather than assumes:

  * TWIN LAW (the gate): a macro at `mux_factor = S` and `S` twin macros at
    `mux_factor = 1`, each holding one slot's logical columns, reach the same
    converged operating point BIT-EXACTLY, column for column, on every solved
    quantity (both rail node profiles, the cell working point, both port
    states). The slot's columns are addressed through the macro's own slot map,
    so the comparison also executes the seat-to-physical scatter that
    `program` performs: a misplaced digit or a mis-scattered seat moves cells
    between physical columns and the per-column match breaks.
  * CODE TWIN: the same commutation end to end — the big macro's logical output
    column `io * S + s` equals the twin macro's column `io`, so program,
    solve, readout, and code assembly all commute with the serialization.
  * PLACEMENT LAW: the slot map is the documented bijection `phys_col =
    ((slot * gn + io) * polarity + pol) * w_digit + digit`, and the two macro
    conversions `_seat_to_phys` / `_phys_to_seat` are mutual inverses on
    both the program layout (a trailing row axis) and the readout layout (a
    trailing column axis).
  * ENCODER LAW: `program` writes LSB-first magnitude digits into the
    `(PWG, NWG)` polarity pair at those physical columns. This one is checked
    against an oracle built outside the macro, because the twin comparisons
    above cannot see it: both twin sides run the SAME encoder, so any
    permutation of the encoder's own axes (digit order, polarity order) cancels
    on both sides of every twin equality.
  * WL-PER-PLANE LAW: the array's cap row is invariant to the column-MUX depth
    at a fixed logical geometry — the WL ladder bills once per solve inside the
    kernel mode function, so no per-slot WL billing can creep back in.
  * TRUE-SHAPE LAW: the macro fabricates its cablc / sl_driver clamp banks at
    `(gn, 2, w_digit)` and its dswct / sinwp_sc / pn_isub readout modules at
    `(gn, 2)` / `(gn, 2)` / `(gn,)`, all derived from the config (no magic
    numbers) and prefix-safe under a fabrication prefix, while the array seats
    every physical column.

Runs eagerly (dynamo disabled) so the `@torch.compile` solver leaf is not
unrolled.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox import Profiler, Reporter
from neurox.primitive.xbar.cell import XbarCell1t1rDcop, XbarCell1t1rLinear
from neurox.primitive.xbar.solver import ColBlColSlDcop, ColBlColSlProber

from ._utils import (
    MAG_MAX,
    QUANTIZATION_MODE,
    TINY_ADC_BITS,
    TINY_INPUT_NUM,
    TINY_OUTPUT_NUM,
    Xue2020JsscCimMacro,
    build_config,
    build_macro,
    midpoint_refs,
    probe_i_sub_grid,
    with_ref_levels,
)

_POLARITY_NUM = 2
# Bit-exact: the twin is a claim about identical arithmetic, not about agreement
# to some tolerance.
_EXACT = {"rtol": 0.0, "atol": 0.0}


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is `@torch.compile`; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


# ---------------------------------------------------------------------------
# Witness pair: one serialized macro and its per-slot twins
# ---------------------------------------------------------------------------


def _mixed_weight(input_num: int, output_num: int) -> Tensor:
    """Deterministic mixed-sign weight spanning both signs, both digits, and zeros.

    Every physical column must differ from its neighbours, otherwise a
    permutation of the columns would be invisible to a per-column comparison.
    """
    gen = torch.Generator().manual_seed(11)
    return torch.randint(-3, 4, (input_num, output_num), generator=gen, dtype=torch.long)


def _twin_pair(
    device: torch.device, *, mux_factor: int = 2, output_num: int = TINY_OUTPUT_NUM
) -> tuple[Xue2020JsscCimMacro, list[Xue2020JsscCimMacro], Tensor]:
    """Build the serialized macro, its per-slot twins, and the weight they share.

    The twins are the same macro at `mux_factor = 1` over `gn` logical
    columns — one column-MUX slot's worth of hardware, with no slot axis left to
    serialize. Slot `s` holds the big macro's logical columns `s::mux_factor`
    (the placement law `col = io * mux_factor + slot`), and both sides share
    one calibrated ladder so the codes are comparable.
    """
    gn = output_num // mux_factor
    config = build_config(mux_factor=mux_factor)
    grid = probe_i_sub_grid(build_macro(config, device=device, output_num=output_num), m_max=MAG_MAX)
    levels = midpoint_refs(grid, adc_bits=TINY_ADC_BITS)

    big = build_macro(with_ref_levels(config, levels), device=device, output_num=output_num)
    twin_config = with_ref_levels(build_config(mux_factor=1), levels)
    twins = [build_macro(twin_config, device=device, output_num=gn) for _ in range(mux_factor)]

    w = _mixed_weight(TINY_INPUT_NUM, output_num)
    big.program(w.to(device))
    for slot, twin in enumerate(twins):
        twin.program(w[:, slot::mux_factor].contiguous().to(device))
    return big, twins, w


def _solve_dcop(macro: Xue2020JsscCimMacro, x: Tensor) -> ColBlColSlDcop[XbarCell1t1rDcop]:
    """Run one VMM and return the converged solver DCOP it produced."""
    # The DCOP stays where it was solved, which is where the macro's own slot
    # map that indexes it lives.
    with ColBlColSlProber(min_outer=0) as probe, torch.no_grad():
        macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
    # One solve closes with one terminal record; the rest of the book is its
    # iteration trajectory.
    dcops = [record.dcop for record in probe.records if record.dcop is not None]
    assert len(dcops) == 1, f"expected one unchunked solve, got {len(dcops)}"
    return dcops[0]


# ---------------------------------------------------------------------------
# Twin law (the P4 gate)
# ---------------------------------------------------------------------------


def test_serialization_commutes_with_the_flattened_solve(device: torch.device) -> None:
    """One full-array solve equals the per-slot solves, column for column, bit-exactly.

    The big macro settles all `mux_factor * gn * 2 * w_digit` physical columns
    in one solve; each twin settles exactly one slot's columns. Reading the big
    macro's converged state at the physical columns its own slot map assigns to
    slot `s` must reproduce the twin's state exactly — on both rail node
    profiles, on the cell working point, and on both port states.
    """
    mux_factor = 2
    big, twins, _w = _twin_pair(device, mux_factor=mux_factor)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0]], dtype=torch.long, device=device)  # batch (2,)

    big_dcop = _solve_dcop(big, x)
    # Shape: [sweep, gn, polarity, w_digit] -> [sweep, act]
    slot_map = big._slot_map.reshape(mux_factor, -1)

    for slot, twin in enumerate(twins):
        twin_dcop = _solve_dcop(twin, x)
        cols = slot_map[slot]
        for field in ("v_bl_node__V", "v_sl_node__V"):
            # Shape: [..., act, row]
            torch.testing.assert_close(
                getattr(big_dcop, field).index_select(-2, cols), getattr(twin_dcop, field), **_EXACT
            )
        for field in ("v_x__V", "i__uA"):
            # Shape: [..., act, row]
            torch.testing.assert_close(
                getattr(big_dcop.cell, field).index_select(-2, cols), getattr(twin_dcop.cell, field), **_EXACT
            )
        for field in ("i_bl_driver__uA", "v_bl_clamp__V", "i_sl_driver__uA", "v_sl_drive__V"):
            # Shape: [..., act]
            torch.testing.assert_close(
                getattr(big_dcop, field).index_select(-1, cols), getattr(twin_dcop, field), **_EXACT
            )

    # Vacuity guard: the columns must actually differ, otherwise any permutation
    # of them would satisfy the comparison above.
    i_bl = big_dcop.i_bl_driver__uA
    assert float(i_bl.max() - i_bl.min()) > 0.0, "witness columns are indistinguishable; the twin proves nothing"


def test_vec_mat_mul_commutes_with_serialization(device: torch.device) -> None:
    """End to end: the big macro's column `io * S + s` is the slot-`s` twin's column `io`.

    Program, solve, readout chain, and code assembly all commute with the
    column-MUX serialization, so the whole pipeline is layout-independent.
    """
    mux_factor = 2
    big, twins, _w = _twin_pair(device, mux_factor=mux_factor)
    x = torch.tensor([[1, 2, 1, 0], [3, 3, 1, 0], [0, 1, 2, 3]], dtype=torch.long, device=device)

    with torch.no_grad():
        out = big.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS).cpu()
        for slot, twin in enumerate(twins):
            twin_out = twin.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS).cpu()
            assert torch.equal(out[..., slot::mux_factor], twin_out), (
                f"slot {slot} decode differs:\n{out[..., slot::mux_factor].tolist()}\nvs\n{twin_out.tolist()}"
            )
    # Vacuity guard: a constant output would satisfy any column mapping.
    assert int(out.min()) < 0 < int(out.max()), f"witness decode is degenerate: {out.tolist()}"


# ---------------------------------------------------------------------------
# Placement law: the slot map and the two conversions
# ---------------------------------------------------------------------------


def test_slot_map_is_the_documented_bijection(device: torch.device) -> None:
    """`phys_col = ((slot * gn + io) * polarity + pol) * w_digit + digit`, a bijection."""
    for w_digit_num, mux_factor in ((2, 2), (3, 2), (1, 4)):
        config = build_config(w_digit_num=w_digit_num, mux_factor=mux_factor)
        macro = build_macro(config, device=device)
        gn = macro.col_num // mux_factor
        phys_col_num = macro.col_num * w_digit_num * _POLARITY_NUM
        want = torch.arange(phys_col_num, device=macro._slot_map.device).reshape(
            mux_factor, gn, _POLARITY_NUM, w_digit_num
        )
        assert torch.equal(macro._slot_map, want)
        # A bijection onto the physical columns: every column seated exactly once.
        assert torch.equal(torch.sort(macro._slot_map.reshape(-1)).values, want.reshape(-1))


@pytest.mark.parametrize("shuffled", [False, True])
def test_seat_and_phys_conversions_are_mutual_inverses(device: torch.device, shuffled: bool) -> None:
    """The two conversions round-trip on both layouts they are used at.

    `program` converts a seat block with a trailing row axis (`dim = -5`);
    the readout converts a trailing column axis back (`dim = -1`). Each
    direction must undo the other, or a solved column would be billed and read
    out under another column's seat.

    The shipped slot map enumerates the seats in physical order, which makes the
    permutation the identity and would hide a dropped gather; the `shuffled`
    case installs a non-trivial bijection on the instance under test, so the
    inverse index is exercised as an index rather than as a no-op.
    """
    config = build_config(w_digit_num=3, mux_factor=2)
    macro = build_macro(config, device=device)
    seat_shape = tuple(macro._slot_map.shape)
    dev = macro._slot_map.device
    if shuffled:
        gen = torch.Generator(device=dev).manual_seed(5)
        shuffle = torch.randperm(math.prod(seat_shape), generator=gen, device=dev)
        macro._slot_map = shuffle.reshape(seat_shape)
        macro._seat_of_phys = torch.argsort(shuffle)

    # Readout layout: [..., phys_col] <-> [..., sweep, gn, polarity, wd].
    phys = torch.arange(math.prod(seat_shape), device=dev).reshape(1, -1)
    torch.testing.assert_close(macro._seat_to_phys(macro._phys_to_seat(phys, dim=-1), dim=-4), phys, **_EXACT)
    # A seat holds the physical column the map assigns it, in seat enumeration order.
    torch.testing.assert_close(macro._phys_to_seat(phys, dim=-1).reshape(-1), macro._slot_map.reshape(-1), **_EXACT)

    # Program layout: the same block with a trailing row axis.
    seat = torch.arange(math.prod(seat_shape) * macro.row_num, device=dev).reshape(*seat_shape, macro.row_num)
    torch.testing.assert_close(macro._phys_to_seat(macro._seat_to_phys(seat, dim=-5), dim=-2), seat, **_EXACT)


# ---------------------------------------------------------------------------
# Encoder law: what the twin comparisons structurally cannot see
# ---------------------------------------------------------------------------


def test_program_writes_lsb_first_digits_at_the_documented_columns(device: torch.device) -> None:
    """`program` seats LSB-first magnitude digits and the `(PWG, NWG)` pair as documented.

    The twin equalities cannot pin this: both sides encode with the same
    `_seat_states`, so swapping the digit order (MSB-first) or the two
    polarity cells permutes both sides identically and cancels. The oracle here
    is built from the documented bijection and the true-form digit definition
    alone — digit `k` carries place value `radix ** k`, `+m` writes the
    PWG cell to state `m` and the NWG cell to HRS, `-m` the reverse — and is
    compared against the conductances the cells actually carry after a real
    `program` call.
    """
    w_digit_num, mux_factor, radix = 2, 2, 2
    config = build_config(w_digit_num=w_digit_num, mux_factor=mux_factor, w_digit_radix=radix)
    macro = build_macro(config, device=device)
    gn = macro.col_num // mux_factor

    # Asymmetric magnitudes (|w| = 1 -> digits (1, 0), |w| = 2 -> (0, 1)) and both
    # signs, so digit order and polarity are both observable.
    w = torch.tensor([[1, -1, 2, -2], [2, -2, 1, -1], [3, 0, -3, 0], [0, 3, 0, -3]], dtype=torch.long)
    assert tuple(w.shape) == (macro.row_num, macro.col_num)

    # Shape: [phys_col, row]
    want_state = torch.zeros(macro.col_num * _POLARITY_NUM * w_digit_num, macro.row_num, dtype=torch.long)
    for row in range(macro.row_num):
        for col in range(macro.col_num):
            value = int(w[row, col])
            io, slot = divmod(col, mux_factor)  # col = io * mux_factor + slot
            for digit in range(w_digit_num):
                magnitude = abs(value) // radix**digit % radix
                for pol in range(_POLARITY_NUM):  # 0 = PWG (positive), 1 = NWG (negative)
                    phys_col = ((slot * gn + io) * _POLARITY_NUM + pol) * w_digit_num + digit
                    carries = (value >= 0) if pol == 0 else (value < 0)
                    want_state[phys_col, row] = magnitude if carries else 0

    # Vacuity guards: the witness must actually move under the two regressions
    # this test exists to catch.
    seat_state = want_state.unflatten(0, (mux_factor, gn, _POLARITY_NUM, w_digit_num))
    assert not torch.equal(seat_state, seat_state.flip(-2)), "witness is blind to an MSB-first digit order"
    assert not torch.equal(seat_state, seat_state.flip(-3)), "witness is blind to a PWG/NWG swap"

    macro.program(w.to(device))
    cell = macro.array.cell
    assert isinstance(cell, XbarCell1t1rLinear)
    # The state index survives only as the branch parameters it selected, so the
    # oracle is compared through the same table (injective on the witness).
    table__uS = cell._g_cell_on_table__uS
    assert len(torch.unique(table__uS)) == table__uS.numel(), "witness conductance table is degenerate"
    torch.testing.assert_close(cell._g_cell_on__uS, table__uS[want_state.to(device)], **_EXACT)


# ---------------------------------------------------------------------------
# WL-per-plane law
# ---------------------------------------------------------------------------


def test_array_cap_energy_independent_of_mux_factor(device: torch.device) -> None:
    """The array's cap row does not move with the column-MUX depth.

    At a fixed logical geometry the physical column set is the same whatever
    `mux_factor` is — only the seat each column sits in changes — and the WL
    ladder bills once per solve, i.e. once per PLANE, inside the kernel mode
    function. A per-slot WL bill (the pre-flattening special case) would scale
    the WL wire and gate terms with `mux_factor` and break this.
    """
    w = _mixed_weight(TINY_INPUT_NUM, TINY_OUTPUT_NUM)
    x = torch.tensor([1, 2, 1, 3], dtype=torch.long, device=device)

    def array_row(mux_factor: int) -> float:
        macro = build_macro(build_config(mux_factor=mux_factor), device=device)
        macro.program(w.to(device))
        with Profiler() as prof, torch.no_grad():
            macro.vec_mat_mul(x, quantization_mode=QUANTIZATION_MODE, adc_bits=TINY_ADC_BITS)
        return Reporter(macro).by_name(prof)["array"]

    # Same 16 physical columns, factored into 2 slots of 8 lanes vs 4 of 4.
    e_mux2 = array_row(2)
    e_mux4 = array_row(4)
    assert e_mux2 > 0.0
    assert e_mux4 == pytest.approx(e_mux2, rel=1e-12), (
        f"array cap energy moved with the MUX factoring: {e_mux2} vs {e_mux4} "
        "(the WL ladder must bill once per plane, independent of mux_factor)"
    )


# ---------------------------------------------------------------------------
# True-shape law
# ---------------------------------------------------------------------------


def test_true_shape_law(device: torch.device) -> None:
    """Clamp banks + readout modules fabricate at the true hardware counts derived from the config."""
    for w_digit_num, mux_factor in ((2, 2), (3, 2)):
        config = build_config(w_digit_num=w_digit_num, mux_factor=mux_factor)
        for inst in ((), (2,)):
            macro = build_macro(config, device=device, inst_shape=inst)
            gn = macro.col_num // config.mux_factor
            lane = (gn, _POLARITY_NUM, config.w_digit_num)
            for name, trailing in (
                ("cablc", lane),
                ("sl_driver", lane),
                ("dswct", (gn, _POLARITY_NUM)),
                ("sinwp_sc", (gn, _POLARITY_NUM)),
                ("pn_isub", (gn,)),
            ):
                module = getattr(macro, name)
                want = (*inst, *trailing)
                assert module.inst_shape == want, f"{name}.inst_shape {module.inst_shape} != {want}"
                assert module.inst_count == math.prod(inst) * math.prod(trailing)
            # The array seats every physical column: col_num = sweep * gn * 2 * wd.
            assert macro.array.weight_grid_shape == (
                *inst,
                config.mux_factor * gn * _POLARITY_NUM * config.w_digit_num,
                macro.row_num,
            )

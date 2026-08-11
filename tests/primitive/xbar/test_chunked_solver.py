"""Laws for the chunking layer between an array and a fixed-shape solver.

Five laws, on a stub solver and a stub measurement that both record exactly
what they were handed:

  * SAME-SIGNATURE LAW: the wrapper's call is the wrapped solver's call —
    every snap argument arrives chunk-sliced, every other argument arrives
    identical (same object) to what the caller passed.
  * MEASURE-ONLY LAW: a measure tensor reaches the measurement sliced by the
    same rule as a snap's fields, and never reaches the wrapped solver, which
    declares no such keyword. It travels as the bare tensor it is.
  * DECLARED-LEADING LAW: the caller states the leading and every sliced
    operand carries it; the wrapped solver sees one fixed chunk shape and the
    tail chunk is padded to it.
  * FOLD LAW: what comes back is the measurement type at the full leading,
    equal to a single whole-leading solve element for element; no solver
    DCOP escapes the chunk loop.
  * PASS-THROUGH LAW: with no leading anywhere, the snaps and measure tensors
    reach their consumers untouched and the single measurement is returned as
    it stands.

A sixth check guards the declaration itself: an operand that does not carry
the stated leading is rejected before any chunk runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor

from neurox.primitive.xbar.solver import ChunkedSolver, Solver, SolverConfig, SolverDcop

_COL = 3
_ROW = 4


@dataclass(frozen=True)
class _CellSnap:
    """Cell-grid snap: a stride-0 word-line drive plus a programmed buffer."""

    v_wl__V: Tensor
    g__uS: Tensor


@dataclass(frozen=True)
class _ClampSnap:
    """Per-column clamp snap with a constant field expanded over the call."""

    v_ref__V: Tensor
    r_out__MOhm: Tensor


@dataclass(frozen=True)
class _CellDcop:
    """Nested cell result."""

    i__uA: Tensor


@dataclass(frozen=True)
class _Measure:
    """What survives one chunk: the port state and one folded row sum."""

    i_port__uA: Tensor
    wl_sum__V: Tensor


class _SpySolver(Solver):
    """Stub solver recording every call and returning a deterministic DCOP."""

    def __init__(self, *, config: SolverConfig | None = None) -> None:
        del config
        self.calls: list[dict[str, Any]] = []

    def solve_dc(self, **kwargs: Any) -> SolverDcop[_CellDcop]:
        self.calls.append(kwargs)
        cell_snap = kwargs["cell_snap"]
        bl_snap = kwargs["bl_driver_snap"]
        # Shape: [..., col, row]
        node = cell_snap.v_wl__V * cell_snap.g__uS + bl_snap.v_ref__V.unsqueeze(-1)
        # Shape: [..., col]
        port = node.sum(dim=-1) + kwargs["bl_segment_r__MOhm"]
        return SolverDcop(
            i_bl_driver=port,
            i_sl_driver=-port,
            v_bl_node=node,
            v_sl_node=node * 0.5,
            cell=_CellDcop(i__uA=node * 2.0),
            v_bl_clamp=bl_snap.v_ref__V + port,
            v_sl_drive=bl_snap.v_ref__V - port,
        )


class _SpyMeasure:
    """Stub measurement recording every per-chunk call it is handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results: list[_Measure] = []

    def __call__(
        self,
        *,
        dcop: SolverDcop[_CellDcop],
        cell_snap: _CellSnap,
        bl_driver_snap: _ClampSnap,
        v_wl__V: Tensor,
    ) -> _Measure:
        self.calls.append({"dcop": dcop, "cell_snap": cell_snap, "bl_driver_snap": bl_driver_snap, "v_wl__V": v_wl__V})
        # Shape: [..., row] -> [...]
        result = _Measure(i_port__uA=dcop.i_bl_driver, wl_sum__V=v_wl__V.sum(dim=-1))
        self.results.append(result)
        return result


def _cell_snap(leading: tuple[int, ...]) -> _CellSnap:
    """One word line per row held across the columns, over a shared cell buffer."""
    return _CellSnap(
        v_wl__V=torch.randn(*leading, 1, _ROW).expand(*leading, _COL, _ROW),
        g__uS=torch.randn(_COL, _ROW).expand(*leading, _COL, _ROW),
    )


def _clamp_snap(leading: tuple[int, ...]) -> _ClampSnap:
    return _ClampSnap(
        v_ref__V=torch.randn(*leading, _COL),
        r_out__MOhm=torch.zeros(()).expand(*leading, _COL),
    )


def _call(
    solver: ChunkedSolver[_Measure],
    leading: tuple[int, ...],
    segment_r: float,
    *,
    measure: _SpyMeasure | None = None,
) -> _Measure:
    torch.manual_seed(5)
    return solver.solve_dc(
        leading=leading,
        bl_segment_r__MOhm=segment_r,
        sl_segment_r__MOhm=2.0 * segment_r,
        cell="cell-module",
        cell_snap=_cell_snap(leading),
        bl_driver="bl-module",
        bl_driver_snap=_clamp_snap(leading),
        measure=measure if measure is not None else _SpyMeasure(),
        measure_tensors={"v_wl__V": torch.randn(*leading, _ROW)},
    )


def test_snaps_are_sliced_and_everything_else_passes_through() -> None:
    inner = _SpySolver()
    segment_r = 1e-4
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(inner, chunk_size=4)

    _call(chunked, (2, 3), segment_r)

    for call in inner.calls:
        # Non-snap arguments arrive as the very objects the caller passed.
        assert call["cell"] == "cell-module"
        assert call["bl_driver"] == "bl-module"
        # A plain float rail constant rides through untouched.
        assert call["bl_segment_r__MOhm"] == segment_r
        assert call["sl_segment_r__MOhm"] == 2.0 * segment_r
        # Snap arguments arrive on one chunk axis, at the same snap types.
        assert type(call["cell_snap"]) is _CellSnap
        assert call["cell_snap"].v_wl__V.shape == (4, _COL, _ROW)
        assert call["bl_driver_snap"].v_ref__V.shape == (4, _COL)
        # A field that is stride-0 over the whole leading keeps a size-1 axis.
        assert call["cell_snap"].g__uS.shape == (1, _COL, _ROW)


def test_measure_tensors_reach_the_measurement_alone() -> None:
    inner = _SpySolver()
    measure = _SpyMeasure()
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(inner, chunk_size=4)

    _call(chunked, (2, 3), 1e-4, measure=measure)

    for call in inner.calls:
        # The wrapped solver declares no such keyword and never sees one.
        assert "v_wl__V" not in call
        assert "measure" not in call
        assert "measure_tensors" not in call
    for call in measure.calls:
        # A bare tensor is sliced by the same rule as a snap's fields, and
        # arrives as a tensor rather than wrapped in a dataclass.
        assert isinstance(call["v_wl__V"], Tensor)
        assert call["v_wl__V"].shape == (4, _ROW)
        assert call["cell_snap"].v_wl__V.shape == (4, _COL, _ROW)
        assert call["bl_driver_snap"].v_ref__V.shape == (4, _COL)


def test_the_caller_states_the_leading_and_the_tail_is_padded() -> None:
    inner = _SpySolver()
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(inner, chunk_size=4)

    _call(chunked, (2, 3), 1e-4)

    # 6 leading positions in chunks of 4: two calls, both at the same shape.
    assert len(inner.calls) == 2
    assert {call["cell_snap"].v_wl__V.shape for call in inner.calls} == {(4, _COL, _ROW)}


def test_folded_measure_equals_one_whole_leading_solve() -> None:
    leading = (2, 3)
    segment_r = 1e-4
    chunked = _call(ChunkedSolver(_SpySolver(), chunk_size=4), leading, segment_r)
    whole = _call(ChunkedSolver(_SpySolver(), chunk_size=0), leading, segment_r)

    assert type(chunked) is _Measure
    assert chunked.i_port__uA.shape == (*leading, _COL)
    assert chunked.wl_sum__V.shape == leading
    torch.testing.assert_close(chunked.i_port__uA, whole.i_port__uA, rtol=0.0, atol=0.0)
    torch.testing.assert_close(chunked.wl_sum__V, whole.wl_sum__V, rtol=0.0, atol=0.0)


def test_an_operand_short_of_the_declared_leading_is_rejected() -> None:
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(_SpySolver(), chunk_size=4)

    with pytest.raises(ValueError, match="bl_driver_snap carries one of shape"):
        chunked.solve_dc(
            leading=(2, 3),
            bl_segment_r__MOhm=1e-4,
            cell_snap=_cell_snap((2, 3)),
            # One leading axis short: the slicer would gather its column axis.
            bl_driver_snap=_clamp_snap((3,)),
            measure=_SpyMeasure(),
            measure_tensors={"v_wl__V": torch.randn(2, 3, _ROW)},
        )


def test_a_measure_tensor_short_of_the_declared_leading_is_rejected() -> None:
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(_SpySolver(), chunk_size=4)

    with pytest.raises(ValueError, match="v_wl__V carries one of shape"):
        chunked.solve_dc(
            leading=(2, 3),
            bl_segment_r__MOhm=1e-4,
            cell_snap=_cell_snap((2, 3)),
            bl_driver_snap=_clamp_snap((2, 3)),
            measure=_SpyMeasure(),
            # A bare tensor is held to the same declaration as a snap field.
            measure_tensors={"v_wl__V": torch.randn(3, _ROW)},
        )


def test_a_call_without_any_leading_reaches_the_solver_untouched() -> None:
    inner = _SpySolver()
    measure = _SpyMeasure()
    chunked: ChunkedSolver[_Measure] = ChunkedSolver(inner, chunk_size=4)

    measured = _call(chunked, (), 1e-4, measure=measure)

    [call] = inner.calls
    assert call["cell_snap"].v_wl__V.shape == (_COL, _ROW)
    assert call["bl_driver_snap"].v_ref__V.shape == (_COL,)
    assert measure.calls[0]["v_wl__V"].shape == (_ROW,)
    assert measured.i_port__uA.shape == (_COL,)
    # Returned as it stands: one chunk, nothing allocated and nothing copied.
    assert measured is measure.results[0]

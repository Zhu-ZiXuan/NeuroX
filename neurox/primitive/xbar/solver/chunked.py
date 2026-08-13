"""Chunking layer between a crossbar array and a fixed-shape DC solver.

See also:
    docs/internals/primitive/xbar/solver/chunked.md
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import walk_tensor_fields

from .base import Solver
from .chunking import MeasureFold, iter_chunks, slice_snap, slice_tensor

MeasureT = TypeVar("MeasureT")


class ChunkedSolver(Generic[MeasureT]):
    """Same-signature `Solver` wrapper that folds one leading chunk by chunk.

    Argument handling is signature-agnostic: a snap argument is chunk-sliced,
    every other argument passes through untouched, and the caller states the
    leading rather than any axis rank being inferred. Each chunk is measured
    down to the small tensors that survive it and those alone are reassembled,
    so a chunk's grid-shaped DCOP never reaches the caller.

    Args:
        solver: Wrapped solver, faithfully solving one fixed shape.
        chunk_size: Leading instances per chunk. A positive value pads each
            chunk to exactly this size so the compiled solver body sees a
            single input shape; `0` solves the whole leading in one block.
    """

    def __init__(self, solver: Solver, *, chunk_size: int) -> None:
        self._solver = solver
        self._chunk_size = chunk_size

    @torch.compiler.disable(
        recursive=False,
        reason="eager chunk loop; the fixed-shape per-chunk solver body is compiled separately",
    )
    def solve_dc(
        self,
        *,
        leading: tuple[int, ...],
        measure: Callable[..., MeasureT],
        measure_tensors: Mapping[str, Tensor] | None = None,
        **kwargs: Any,
    ) -> MeasureT:
        """Solve the whole call by chunk and fold each chunk down to `measure`.

        Args:
            leading: Broadcast-leading shape of the call, which every snap
                tensor field and every measure tensor carries in front of its
                own trailing block.
            measure: Per-chunk measurement, called with `dcop` (the chunk's
                solver return) plus every sliced snap and measure tensor under
                its own keyword, and returning a frozen dataclass of tensors
                that each carry the chunk axis first.
            measure_tensors: Boundary quantities the measurement reads but the
                wrapped solver does not declare. They are sliced by the same
                rule as a snap's fields and reach `measure` alone.
            kwargs: The wrapped solver's own keyword arguments.

        Returns:
            The measurement type, its tensor fields at `[*leading, *trailing]`.

        Raises:
            ValueError: The leading has an empty extent, so no chunk exists
                to allocate the result from.
        """
        solver_snaps = {name: value for name, value in kwargs.items() if _is_snap(value)}
        extra = dict(measure_tensors or {})
        device = _check_leading(leading, {**solver_snaps, **extra})

        fold: MeasureFold[MeasureT] | None = None
        for spec in iter_chunks(leading=leading, chunk_size=self._chunk_size, device=device):
            sliced = {
                name: slice_snap(snap, coords=spec.multi_coords, leading=leading) for name, snap in solver_snaps.items()
            }
            sliced_extra = {
                name: slice_tensor(value, coords=spec.multi_coords, leading=leading) for name, value in extra.items()
            }
            dcop = self._solver.solve_dc(**{**kwargs, **sliced})
            measured = measure(dcop=dcop, **sliced, **sliced_extra)
            if not leading:
                return measured
            if fold is None:
                fold = MeasureFold(measured, b_total=math.prod(leading))
            fold.write(measured, flat_idx=spec.flat_global_idx)
            # This chunk's state dies here, before the next chunk allocates its own.
            del dcop, measured, sliced, sliced_extra
        if fold is None:
            raise ValueError(f"require: a leading with no empty extent; got {leading}")
        return fold.result(leading=leading)


def _is_snap(value: object) -> bool:
    """Tell a snap argument from a pass-through one.

    A snap is a frozen dataclass of tensor fields; every other argument is a
    module, a protocol object, or a constant that carries no leading.
    """
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _operand_tensors(node: object) -> Iterator[Tensor]:
    """Yield a bare tensor, or every tensor field of one snap.

    Dataclass-valued fields recurse; every other field is skipped, which is
    how an absent optional field stays absent.
    """
    if isinstance(node, Tensor):
        yield node
        return
    collected: list[Tensor] = []

    def _record(t: Tensor) -> Tensor:
        collected.append(t)
        return t

    walk_tensor_fields(node, _record)
    yield from collected


def _check_leading(leading: tuple[int, ...], operands: Mapping[str, object]) -> torch.device:
    """Verify every sliced operand carries the call's leading, and read the device.

    The slicer takes the first `len(leading)` axes of a tensor to be the
    leading, so one arriving at anything narrower would have its own trailing
    block gathered instead.

    Args:
        leading: Broadcast-leading shape the caller states.
        operands: Snaps and bare tensors the chunk loop slices, keyed by the
            keyword each arrived under.

    Returns:
        Device the operands' tensors live on, which the chunk coordinates
        are built on.

    Raises:
        ValueError: A tensor does not carry `leading`, or no operand carries a
            tensor at all.
    """
    device: torch.device | None = None
    for name, operand in operands.items():
        for tensor in _operand_tensors(operand):
            if tuple(tensor.shape[: len(leading)]) != leading:
                raise ValueError(
                    f"require: every sliced tensor at the call's leading {leading}; "
                    f"{name} carries one of shape {tuple(tensor.shape)}"
                )
            if device is None:
                device = tensor.device
    if device is None:
        raise ValueError("require: at least one snap field or measure tensor to read the device from")
    return device

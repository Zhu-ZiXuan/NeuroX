"""Memory-bounded execution over a declared leading shape.

The executor partitions a tensor-carrying operand tree into fixed-size
chunks, calls one injected function on each, and folds its result tree back
into the original leading shape.

See Also:
    docs/system_design/xbar_solve.md
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch
from torch import Tensor

from neurox.common import torch_compiler_disable, walk_tensor_fields


class _Chunk(NamedTuple):
    """Coordinates and result interval for one fixed-size call."""

    coords: tuple[Tensor, ...]
    start: int
    stop: int


def _iter_chunks(
    *,
    leading_shape: tuple[int, ...],
    chunk_size: int,
    reference: Tensor,
) -> Iterator[_Chunk]:
    """Yield fixed-size coordinates while keeping only one chunk resident."""
    total = math.prod(leading_shape)
    actual_chunk_size = min(chunk_size, total)
    offsets = torch.arange(actual_chunk_size, device=reference.device)

    for start in range(0, total, actual_chunk_size):
        stop = min(start + actual_chunk_size, total)
        flat = offsets + start
        if stop - start < actual_chunk_size:
            flat = flat.clamp_max(total - 1)
        coords = tuple(torch.unravel_index(flat, leading_shape))
        del flat
        yield _Chunk(
            coords=coords,
            start=start,
            stop=stop,
        )


def _slice_tensor_tree[NodeT](
    node: NodeT,
    *,
    coords: tuple[Tensor, ...],
    leading_shape: tuple[int, ...],
) -> NodeT:
    """Select the same leading positions from every tensor in a dataclass tree."""
    return walk_tensor_fields(node, lambda t: _slice_tensor(t, coords=coords, leading_shape=leading_shape))


def _slice_tensor(
    tensor: Tensor,
    *,
    coords: tuple[Tensor, ...],
    leading_shape: tuple[int, ...],
) -> Tensor:
    """Select one chunk without materialising stride-0 broadcasts in full."""
    leading_rank = len(leading_shape)

    # --- 1: collapse trailing axes the tensor only broadcasts over ---

    view = tensor
    trailing_shape = tuple(tensor.shape[leading_rank:])
    collapsed = False
    for axis in range(leading_rank, tensor.ndim):
        if view.size(axis) > 1 and view.stride(axis) == 0:
            view = view.narrow(axis, 0, 1)
            collapsed = True

    # --- 2: replace all leading axes with one chunk axis ---

    index: list[int | Tensor] = [0 if view.stride(axis) == 0 else coords[axis] for axis in range(leading_rank)]
    sliced = view[tuple(index)]
    if not any(isinstance(entry, Tensor) for entry in index):
        # Every leading axis shares one value, so indexing leaves no chunk
        # axis. Restore a size-1 axis that broadcasts against varying inputs.
        sliced = sliced.unsqueeze(0)

    # --- 3: restore collapsed trailing axes as stride-0 views ---

    if collapsed:
        sliced = sliced.expand(*sliced.shape[: sliced.ndim - len(trailing_shape)], *trailing_shape)
    return sliced


class _ResultFold[ResultT]:
    """Full-leading buffers filled by consecutive per-chunk results."""

    def __init__(self, first: ResultT, *, total: int) -> None:
        self._targets: list[Tensor] = []

        def allocate(value: Tensor) -> Tensor:
            target = torch.empty(total, *value.shape[1:], dtype=value.dtype, device=value.device)
            self._targets.append(target)
            return target

        self._folded = walk_tensor_fields(first, allocate)

    def write(self, chunk: ResultT, *, start: int, stop: int) -> None:
        """Write the valid prefix of one padded result into its interval."""
        positions = iter(self._targets)
        valid_size = stop - start

        def write(value: Tensor) -> Tensor:
            next(positions)[start:stop] = value[:valid_size]
            return value

        walk_tensor_fields(chunk, write)

    def result(self, *, leading_shape: tuple[int, ...]) -> ResultT:
        """Restore the original leading shape on every folded tensor."""

        def unflatten(value: Tensor) -> Tensor:
            return value.reshape(*leading_shape, *value.shape[1:])

        return walk_tensor_fields(self._folded, unflatten)


@torch_compiler_disable(
    recursive=False,
    reason="eager chunk loop; the fixed-shape per-chunk run body compiles separately",
)
def execute_chunked[OperandsT, ResultT](
    *,
    chunk_size: int,
    leading_shape: tuple[int, ...],
    operands: OperandsT,
    run: Callable[[OperandsT], ResultT],
) -> ResultT:
    """Run an operand tree by fixed-size chunk and restore its leading shape.

    Args:
        chunk_size: Maximum leading positions per chunk. A positive value uses
            the smaller of this bound and the complete leading extent, then
            pads every tail chunk to that effective size; `0` passes the
            complete operand tree through.
        leading_shape: Exact prefix carried by every operand tensor.
        operands: Full-leading dataclass tensor tree.
        run: Function from one sliced operand tree to a result tree that
            preserves the sliced leading axis.

    Returns:
        The result tree with every tensor restored to `leading_shape`.
    """
    reference = _check_leading(leading_shape, operands)
    if chunk_size == 0 or not leading_shape:
        return run(operands)

    total = math.prod(leading_shape)
    chunks = _iter_chunks(
        leading_shape=leading_shape,
        chunk_size=chunk_size,
        reference=reference,
    )

    first = next(chunks)
    sliced = _slice_tensor_tree(operands, coords=first.coords, leading_shape=leading_shape)
    first_result = run(sliced)
    fold = _ResultFold(first_result, total=total)
    fold.write(first_result, start=first.start, stop=first.stop)
    del first_result, sliced

    for chunk in chunks:
        sliced = _slice_tensor_tree(operands, coords=chunk.coords, leading_shape=leading_shape)
        result = run(sliced)
        fold.write(result, start=chunk.start, stop=chunk.stop)
        del result, sliced

    return fold.result(leading_shape=leading_shape)


def _operand_tensors(node: object) -> list[Tensor]:
    """Collect every tensor field of a dataclass operand tree."""
    tensors: list[Tensor] = []

    def collect(tensor: Tensor) -> Tensor:
        tensors.append(tensor)
        return tensor

    walk_tensor_fields(node, collect)
    return tensors


def _check_leading(leading_shape: tuple[int, ...], operands: object) -> Tensor:
    """Verify the declared prefix once and return a reference tensor."""
    tensors = _operand_tensors(operands)
    for tensor in tensors:
        if tuple(tensor.shape[: len(leading_shape)]) != leading_shape:
            raise ValueError(
                f"require: every operand tensor at the call's leading {leading_shape}; "
                f"got one of shape {tuple(tensor.shape)}"
            )
    return tensors[0]

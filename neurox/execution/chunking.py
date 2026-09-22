"""Memory-bounded execution over a declared leading shape."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from neurox.common.dataclass_mixin import (
    map_paired_tensor_fields,
    map_single_tensor_fields,
)

from .loop import run_map

if TYPE_CHECKING:
    from _typeshed import DataclassInstance as _Dataclass

__all__ = ["run_chunked"]


def run_chunked[InputsT: _Dataclass, OutputsT: _Dataclass](
    *,
    expected_chunk_size: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    operands: InputsT,
    output_template: OutputsT,
    body_fn: Callable[[InputsT], OutputsT],
) -> OutputsT:
    """Evaluate independent positional chunks and reassemble their outputs.

    Equal-sized chunks run in at most two maps; singleton groups run directly.
    Only real positions enter each callback. Expanded inputs are gathered on
    demand, and no state is carried between chunks. Complete outputs,
    including histories, occupy storage proportional to the full leading
    extent; joining groups can temporarily retain both groups and their result.

    Equivalent Python control flow:

    ```python
    size = math.prod(leading_shape)
    count = 1 if expected_chunk_size == 0 else math.ceil(size / expected_chunk_size)
    small, extra = divmod(size, count)
    sizes = [small + 1] * extra + [small] * (count - extra)
    flat_inputs = operands.reshape(size, *operands.shape[len(leading_shape):])
    outputs = []
    start = 0
    for chunk_size in sizes:
        inputs = flat_inputs.narrow(0, start, chunk_size)
        outputs.append(body_fn(inputs))
        start += chunk_size
    output = torch.cat(outputs, dim=0)
    return output.reshape((*leading_shape, *output.shape[1:]))
    ```

    Args:
        expected_chunk_size: Positive upper bound on positions per chunk;
            selects the fewest balanced chunks. Zero selects one unbounded chunk.
        leading_shape: Position prefix of input and assembled output tensors.
            Inputs may be expanded views with zero strides. Extents are
            positive; an empty tuple is one position.
        device: Placement shared by input tensors, output tensors, and indices.
        operands: Dataclass instance, optionally containing nested dataclasses.
            Tensor fields are sliced; other fields remain unchanged. Dataclasses
            support reconstruction through `dataclasses.replace`.
        output_template: Result dataclass structure used by map, including
            nested dataclasses and optional fields. Map requires registered
            PyTrees with tensor leaves; template values and metadata are unused.
        body_fn: Receives a dataclass whose tensor fields have one leading
            chunk axis; returns a dataclass with that exact chunk axis.
            Mapped outputs match `output_template`; remaining dimensions stay fixed.
            Depends only on its input chunk and captured constants.

    Returns:
        Results in row-major position order with the chunk axis replaced by
        `leading_shape`. Trailing dimensions and optional fields are preserved.

    Raises:
        ValueError: The chunk bound is negative or there are no positions.
    """
    # --- 1: validate the schedule ---

    if expected_chunk_size < 0:
        raise ValueError(f"expected_chunk_size must be nonnegative; got {expected_chunk_size}")

    leading_size = math.prod(leading_shape)
    if leading_size == 0:
        raise ValueError("leading_shape must contain at least one position")

    # --- 2: construct the balanced static schedule ---

    chunk_num = 1 if expected_chunk_size == 0 else math.ceil(leading_size / expected_chunk_size)
    chunk_size = math.ceil(leading_size / chunk_num)
    smaller_chunk_num = chunk_num * chunk_size - leading_size
    larger_chunk_num = chunk_num - smaller_chunk_num

    # --- 3: execute the two uniform groups in position order ---

    result = _run_uniform_group(
        group_chunk_size=chunk_size,
        group_chunk_num=larger_chunk_num,
        start_offset=0,
        leading_shape=leading_shape,
        device=device,
        operands=operands,
        output_template=output_template,
        body_fn=body_fn,
    )
    if smaller_chunk_num > 0:
        smaller_result = _run_uniform_group(
            group_chunk_size=chunk_size - 1,
            group_chunk_num=smaller_chunk_num,
            start_offset=larger_chunk_num * chunk_size,
            leading_shape=leading_shape,
            device=device,
            operands=operands,
            output_template=output_template,
            body_fn=body_fn,
        )
        result = map_paired_tensor_fields(lambda left, right: torch.cat((left, right), dim=0), result, smaller_result)

    # --- 4: restore the caller's leading axes ---

    return _restore_result(result, leading_shape=leading_shape)


def _run_uniform_group[InputsT: _Dataclass, OutputsT: _Dataclass](
    *,
    group_chunk_size: int,
    group_chunk_num: int,
    start_offset: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    operands: InputsT,
    output_template: OutputsT,
    body_fn: Callable[[InputsT], OutputsT],
) -> OutputsT:
    def evaluate_chunk(flat_indices: Tensor) -> OutputsT:
        coords = tuple(torch.unravel_index(flat_indices, leading_shape))
        chunk_operands = _slice_operands(
            operands,
            coords=coords,
            chunk_size=flat_indices.shape[0],
            leading_shape=leading_shape,
        )
        return body_fn(chunk_operands)

    offsets = torch.arange(group_chunk_size, device=device)

    if group_chunk_num == 1:
        return evaluate_chunk(offsets + start_offset)

    starts = start_offset + torch.arange(group_chunk_num, device=device) * group_chunk_size

    def map_body_fn(start: Tensor) -> OutputsT:
        return evaluate_chunk(offsets + start)

    stacked = run_map(xs=starts, body_fn=map_body_fn, output_template=output_template)
    # Shape: [chunk, position, ...] -> [chunk*position, ...]
    return map_single_tensor_fields(lambda tensor: tensor.flatten(0, 1), stacked)


def _restore_result[OutputsT: _Dataclass](
    result: OutputsT,
    *,
    leading_shape: tuple[int, ...],
) -> OutputsT:
    def fn(tensor: Tensor) -> Tensor:
        # Shape: [position, ...] -> [*leading_shape, ...]
        return tensor.reshape((*leading_shape, *tensor.shape[1:]))

    return map_single_tensor_fields(fn, result)


def _slice_operands[InputsT: _Dataclass](
    operands: InputsT,
    *,
    coords: tuple[Tensor, ...],
    chunk_size: int,
    leading_shape: tuple[int, ...],
) -> InputsT:
    leading_rank = len(leading_shape)

    def fn(tensor: Tensor) -> Tensor:
        view = tensor
        trailing_shape = tuple(tensor.shape[leading_rank:])
        # Gather only stored values of broadcast tails, then restore their extents.
        # Otherwise advanced indexing materializes the entire expanded tail.
        for axis in range(leading_rank, tensor.ndim):
            if view.size(axis) > 1 and view.stride(axis) == 0:
                view = view.narrow(axis, 0, 1)

        index = tuple(0 if view.stride(axis) == 0 else coords[axis] for axis in range(leading_rank))
        sliced = view[index]
        if not any(isinstance(entry, Tensor) for entry in index):
            sliced = sliced.unsqueeze(0)
        return sliced.expand(chunk_size, *trailing_shape)

    return map_single_tensor_fields(fn, operands)

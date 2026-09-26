"""While loops, scans, and maps over structured states and inputs.

Equivalent Python control flow blocks are pseudocode. Tensor notation stands
for the same operation on every tensor in a structured value; tree traversal
is omitted.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from neurox.common.torch_compat import torch_map, torch_scan, torch_while_loop

__all__ = [
    "run_map",
    "run_scan",
    "run_scan_without_inputs",
    "run_scan_without_carry",
    "run_scan_without_output",
    "run_while_loop",
    "run_while_loop_with_counter",
]


def run_while_loop[StateT](
    *,
    init_state: StateT,
    cond_fn: Callable[[StateT], Tensor | bool],
    body_fn: Callable[[StateT], StateT],
) -> StateT:
    """Update a structured state while its condition holds.

    Equivalent Python control flow:

    ```python
    state = init_state
    while cond_fn(state):
        state = body_fn(state)
    return state
    ```

    The condition is checked before each update. State PyTree structure and
    carry tensor metadata must remain unchanged across updates.

    Args:
        init_state: Initial structured state whose tensor metadata is retained
            across updates.
        cond_fn: Receives the current state and returns a scalar boolean
            condition deciding whether to execute another update.
        body_fn: Callback mapping the current state to the next state without
            input mutation.

    Returns:
        Terminal state, or `init_state` if the initial condition is false.
    """
    return torch_while_loop(cond_fn, body_fn, init_state)


def run_while_loop_with_counter[StateT](
    *,
    init_state: StateT,
    cond_fn: Callable[[Tensor, StateT], Tensor | bool],
    body_fn: Callable[[Tensor, StateT], StateT],
    device: torch.device | None = None,
) -> StateT:
    """Run a counted while loop over a PyTree state.

    Equivalent Python control flow, with grad mode preserved:

    ```python
    step = torch.zeros((), dtype=torch.int64, device=device)
    state = init_state
    while cond_fn(step, state):
        state = body_fn(step, state)
        step = step + 1
    return state
    ```

    Args:
        init_state: Initial structured state whose tensor metadata is retained
            across updates.
        cond_fn: Receives the zero-based scalar step and current state; decides
            whether another update runs.
        body_fn: Receives the same step and state, returning the next state.
        device: Optional step placement; omission uses the default device.

    Returns:
        Terminal state, or `init_state` if the initial condition is false.
    """

    def loop_cond_fn(carry: tuple[Tensor, StateT]) -> Tensor | bool:
        step, state = carry
        return cond_fn(step, state)

    def loop_body_fn(carry: tuple[Tensor, StateT]) -> tuple[Tensor, StateT]:
        step, state = carry
        next_state = body_fn(step, state)
        return step + 1, next_state

    initial_carry = (torch.zeros((), dtype=torch.int64, device=device), init_state)
    _, final_state = torch_while_loop(loop_cond_fn, loop_body_fn, initial_carry)
    return final_state


def run_scan[StateT, InputT, OutputT](
    *,
    init_state: StateT,
    xs: InputT,
    body_fn: Callable[[StateT, InputT], tuple[StateT, OutputT]],
    output_template: OutputT,
    dim: int = 0,
    reverse: bool = False,
) -> tuple[StateT, OutputT]:
    """Scan PyTree input slices while carrying a state.

    Equivalent Python control flow:

    ```python
    axis = dim if dim >= 0 else dim + xs.ndim
    length = xs.shape[axis]
    indices = range(length - 1, -1, -1) if reverse else range(length)
    state = init_state
    outputs = [None] * length
    for i in indices:
        state, outputs[i] = body_fn(state, xs.select(axis, i))
    output_axis = axis if axis <= outputs[0].ndim else 0
    return state, torch.stack(outputs, dim=output_axis)
    ```

    Args:
        init_state: Initial structured state whose tensor metadata is retained
            across updates.
        xs: PyTree sliced along `dim`.
        body_fn: Receives the current state and one input slice; returns the
            next state and one output slice.
        output_template: Output PyTree structure, including optional fields.
            Tensor values and metadata are unused.
        dim: Input iteration axis; negative values are resolved against the
            first input leaf.
        reverse: Visit inputs in reverse order while returning outputs in input
            order.

    Returns:
        A tuple (state, outputs) containing the final state and stacked outputs.
        Terminal state and stacked outputs. Each output leaf gains an iteration
        axis at the resolved `dim` if that axis exists in the stacked tensor,
        otherwise at axis zero.
    """
    return torch_scan(
        body_fn,
        init_state,
        xs,
        dim=dim,
        reverse=reverse,
        output_template=output_template,
    )


def run_scan_without_output[StateT, InputT](
    *,
    init_state: StateT,
    xs: InputT,
    body_fn: Callable[[StateT, InputT], StateT],
    device: torch.device | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> StateT:
    """Scan PyTree input slices and retain only the terminal state.

    Equivalent Python control flow:

    ```python
    axis = dim if dim >= 0 else dim + xs.ndim
    length = xs.shape[axis]
    indices = range(length - 1, -1, -1) if reverse else range(length)
    state = init_state
    for i in indices:
        state = body_fn(state, xs.select(axis, i))
    return state
    ```

    Args:
        init_state: Initial structured state whose tensor metadata is retained
            across updates.
        xs: PyTree sliced along `dim`.
        body_fn: Receives the current state and one input slice; returns the
            next state.
        device: Optional empty-output placement; omission uses the default
            device.
        dim: Input iteration axis; negative values are resolved against the
            first input leaf.
        reverse: Visit input slices in reverse order.

    Returns:
        Terminal carry with the same structured layout as init_state.
    """

    # Scan requires an output leaf; a zero-length tensor carries no observations.
    def scan_body(state: StateT, input: InputT) -> tuple[StateT, Tensor]:
        return body_fn(state, input), torch.empty(0, device=device)

    state, _ = torch_scan(
        scan_body,
        init_state,
        xs,
        dim=dim,
        reverse=reverse,
        output_template=torch.empty(0, device=device),
    )
    return state


def run_scan_without_inputs[StateT, OutputT](
    *,
    init_state: StateT,
    length: int,
    body_fn: Callable[[StateT], tuple[StateT, OutputT]],
    output_template: OutputT,
    device: torch.device | None = None,
    reverse: bool = False,
) -> tuple[StateT, OutputT]:
    """Repeat a PyTree state update and collect its outputs.

    Equivalent Python control flow:

    ```python
    state = init_state
    outputs = []
    for _ in range(length):
        state, output = body_fn(state)
        outputs.append(output)
    if reverse:
        outputs.reverse()
    return state, torch.stack(outputs, dim=0)
    ```

    Args:
        init_state: Initial structured state whose tensor metadata is retained
            across updates.
        length: Positive number of callback evaluations.
        body_fn: Receives the current state; returns the next state and one
            output slice.
        output_template: Output PyTree structure, including optional fields.
            Tensor values and metadata are unused.
        device: Optional placement of the unused iteration indices; omission
            uses the default device.
        reverse: Reverse output positions; the state recurrence is unchanged.

    Returns:
        A tuple containing:
            Terminal state and outputs with a new leading axis of size `length`.
    """

    def scan_body(state: StateT, _input: Tensor) -> tuple[StateT, OutputT]:
        return body_fn(state)

    steps = torch.arange(length, device=device)
    return torch_scan(
        scan_body,
        init_state,
        steps,
        reverse=reverse,
        output_template=output_template,
    )


def run_scan_without_carry[InputT, OutputT](
    *,
    xs: InputT,
    body_fn: Callable[[InputT], OutputT],
    output_template: OutputT,
    device: torch.device | None = None,
    dim: int = 0,
    reverse: bool = False,
) -> OutputT:
    """Apply a callback independently to PyTree input slices and stack its outputs.

    Equivalent Python control flow:

    ```python
    axis = dim if dim >= 0 else dim + xs.ndim
    length = xs.shape[axis]
    indices = range(length - 1, -1, -1) if reverse else range(length)
    outputs = [None] * length
    for i in indices:
        outputs[i] = body_fn(xs.select(axis, i))
    output_axis = axis if axis <= outputs[0].ndim else 0
    return torch.stack(outputs, dim=output_axis)
    ```

    Args:
        xs: PyTree sliced along `dim`.
        body_fn: Receives one input slice and returns one output slice.
        output_template: Output PyTree structure, including optional fields.
            Tensor values and metadata are unused.
        device: Optional empty-carry placement; omission uses the default device.
        dim: Input iteration axis; negative values are resolved against the
            first input leaf.
        reverse: Visit inputs in reverse order while returning outputs in
            input order.

    Returns:
        Stacked outputs. Each leaf gains an iteration axis at the resolved
        `dim` if that axis exists in the stacked tensor, otherwise at axis zero.
    """

    # A fresh empty carry satisfies scan without aliasing its input carry.
    def scan_body(_state: Tensor, input: InputT) -> tuple[Tensor, OutputT]:
        return torch.empty(0, device=device), body_fn(input)

    _, output = torch_scan(
        scan_body,
        torch.empty(0, device=device),
        xs,
        dim=dim,
        reverse=reverse,
        output_template=output_template,
    )
    return output


def run_map[InputT, OutputT](
    *,
    xs: InputT,
    body_fn: Callable[[InputT], OutputT],
    output_template: OutputT,
) -> OutputT:
    """Evaluate independent leading-axis input slices and stack their outputs.

    Equivalent Python control flow:

    ```python
    outputs = []
    for i in range(xs.shape[0]):
        outputs.append(body_fn(xs.select(0, i)))
    return torch.stack(outputs, dim=0)
    ```

    Args:
        xs: PyTree sliced along axis 0; Tensor leaves share a positive
            leading length.
        body_fn: Receives one input slice and returns one output slice.
            Depends only on that slice and captured constants.
        output_template: Output PyTree structure, including optional fields;
            tensor values and metadata are unused. Each callback output
            must match this structure.

    Returns:
        Outputs in input order, with a new leading axis for the mapped slices.
        Registered dataclasses and their optional fields are preserved.
    """
    return torch_map(body_fn, xs, output_template=output_template)

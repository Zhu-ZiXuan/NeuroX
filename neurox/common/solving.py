"""Convergence-controlled solving and raw observation collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, final

import torch
from torch import Tensor

from .loop import run_scan_without_inputs, run_while_loop_with_counter
from .pytree_dataclass_mixin import PyTreeDataClassMixin
from .tensor_dataclass_mixin import TensorDataClassMixin, walk_single_tensor_fields
from .torch_compat import torch_assert_async, torch_cond

__all__ = ["SolvingState", "SolvingTrace", "run_solving_loop", "run_solving_trace_scan"]


class SolvingState(TensorDataClassMixin, PyTreeDataClassMixin):
    """Registered numerical state whose positions report whether they remain active.

    Excluded positions are inactive. Numerical failures raise rather than
    become inactive. Tensor leaves share a device.
    """

    is_active: Tensor
    """Boolean mask of positions requiring evaluation in this invocation.
    Shape: `[...]`."""

    @property
    def device(self) -> torch.device:
        """Device shared by all tensor leaves."""
        return self.is_active.device

    @property
    def any_active(self) -> Tensor:
        """Whether any positions remain active."""
        return self.is_active.any()


class SolvingTrace(TensorDataClassMixin, PyTreeDataClassMixin):
    """Registered observation or history with trailing iteration axes.

    A history has the same concrete type as one observation. Each enclosing
    scan appends one iteration axis to every tensor, including nested histories.
    Floating observations use NaN for unselected or inactive positions and
    unused steps; boolean flags use false and signed integers use `-1` for
    unused steps. Optional fields remain fixed throughout an invocation.
    """

    @final
    def select(self, index: int) -> Self:
        """Remove the last iteration axis to retrieve one complete observation."""
        return walk_single_tensor_fields(lambda tensor: tensor.select(-1, index), self)

    @final
    def mask_invalid(self, valid: Tensor) -> Self:
        """Replace invalid observations with unused values without changing the input.

        Args:
            valid: Boolean position mask, including singleton broadcast axes.
                Trailing singleton axes are appended to match each tensor's
                rank, including nested traces, before applying the mask.

        Returns:
            Trace with invalid floating values replaced by NaN, booleans by
            false, and signed integers by `-1`. Optional fields are preserved.

        Raises:
            TypeError: A field cannot represent its unused value.
        """

        def mask_tensor(tensor: Tensor) -> Tensor:
            broadcast_valid = valid.reshape((*valid.shape, *((1,) * (tensor.ndim - valid.ndim))))
            unused = False if tensor.dtype == torch.bool else torch.nan if tensor.is_floating_point() else -1
            return torch.where(broadcast_valid, tensor, unused)

        return walk_single_tensor_fields(mask_tensor, self)


def run_solving_loop[StateT: SolvingState](
    *,
    init_state: StateT,
    body_fn: Callable[[StateT], StateT],
    max_iter: int,
    strict: bool,
) -> StateT:
    """Update active positions until convergence or an iteration cap.

    Equivalent Python control flow, omitting entry validation:

    ```python
    state = init_state
    step = 0
    while step < max_iter and state.any_active:
        state = body_fn(state)
        step += 1
    if strict and state.any_active:
        raise RuntimeError("Solving did not converge")
    return state
    ```

    Args:
        body_fn: Returns the next state; owns numerical updates, preservation
            of inactive positions, and numerical validation.
        max_iter: Positive upper bound on callback evaluations.
        strict: Require the terminal state to have no active positions.

    Returns:
        Terminal state, including unconverged positions when `strict=False`.
        An initially inactive state is returned without invoking `body_fn`.

    Raises:
        TypeError: The initial activity mask is not boolean.
        ValueError: `max_iter` is not positive.
        RuntimeError: The terminal state remains active with `strict=True`.
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive; got {max_iter}")
    if init_state.is_active.dtype != torch.bool:
        raise TypeError("is_active must be a boolean mask")

    def solving_cond_fn(step: Tensor, current: StateT) -> Tensor:
        return (step < max_iter) & current.any_active

    def solving_body_fn(_step: Tensor, current: StateT) -> StateT:
        return body_fn(current)

    state = run_while_loop_with_counter(
        init_state=init_state,
        cond_fn=solving_cond_fn,
        body_fn=solving_body_fn,
        device=init_state.device,
    )

    if strict:
        torch_assert_async(~state.any_active, f"Solving did not converge within {max_iter} iterations")

    return state


def run_solving_trace_scan[StateT: SolvingState, TraceT: SolvingTrace](
    *,
    init_state: StateT,
    body_fn: Callable[[StateT], tuple[StateT, TraceT]],
    default_trace: TraceT,
    max_iter: int,
    strict: bool,
    trace_mask: Tensor | None = None,
) -> tuple[StateT, TraceT]:
    """Collect a fixed-capacity history of convergence observations.

    Observations use activity before each update, retaining the final
    convergence check for each position. Selection affects recording only.
    Once all positions are inactive, remaining steps use `default_trace`.

    Equivalent Python control flow:

    ```python
    state = init_state
    observations = []
    for _ in range(max_iter):
        if state.any_active:
            valid = state.is_active
            if trace_mask is not None:
                valid = valid & trace_mask
            state, trace = body_fn(state)
            for _ in range(trace.ndim - valid.ndim):
                valid = valid.unsqueeze(-1)
            trace = torch.where(valid, trace, torch.nan)
        else:
            trace = default_trace
        observations.append(trace)
    if strict and state.any_active:
        raise RuntimeError("Solving did not converge")
    return state, torch.stack(observations, dim=-1)
    ```

    Args:
        body_fn: Returns the next state and one raw observation; owns
            preservation of inactive positions and numerical validation.
        default_trace: Observation filled with unused values, matching the
            masked callback output, including nested histories. The active
            branch produces contiguous observations.
        max_iter: Positive history capacity and upper bound on updates.
        strict: Require the terminal state to have no active positions.
        trace_mask: Optional boolean selection broadcastable to
            `init_state.is_active`, independent of numerical activity.

    Returns:
        Terminal state and history. Each trace tensor gains a final axis of
        size `max_iter`; existing nested history axes keep their order.
        Unconverged positions are retained when `strict=False`.

    Raises:
        TypeError: The activity mask is not boolean, or an observation field
            cannot represent its unused value.
        RuntimeError: The terminal state remains active with `strict=True`.
    """
    if init_state.is_active.dtype != torch.bool:
        raise TypeError("is_active must be a boolean mask")

    def active_body_fn(current: StateT) -> tuple[StateT, TraceT]:
        next_state, trace = body_fn(current)
        valid = current.is_active if trace_mask is None else current.is_active & trace_mask
        trace = trace.mask_invalid(valid)
        # Explicit format also fixes size-one strides across cond branches.
        trace = walk_single_tensor_fields(lambda tensor: tensor.clone(memory_format=torch.contiguous_format), trace)
        return next_state, trace

    def inactive_body_fn(current: StateT) -> tuple[StateT, TraceT]:
        return (
            walk_single_tensor_fields(torch.clone, current),
            walk_single_tensor_fields(torch.clone, default_trace),
        )

    def solving_body_fn(current: StateT) -> tuple[StateT, TraceT]:
        return torch_cond(
            current.any_active,
            active_body_fn,
            inactive_body_fn,
            current,
            output_template=(current, default_trace),
        )

    state, trace = run_scan_without_inputs(
        init_state=init_state,
        length=max_iter,
        body_fn=solving_body_fn,
        output_template=default_trace,
        device=init_state.device,
    )
    trace = walk_single_tensor_fields(lambda tensor: tensor.movedim(0, -1), trace)

    if strict:
        torch_assert_async(~state.any_active, f"Solving did not converge within {max_iter} iterations")

    return state, trace

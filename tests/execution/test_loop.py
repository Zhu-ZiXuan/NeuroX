"""Counted loops pass the current step to conditions and updates."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.dataclass_mixin import PyTreeDataClassMixin, TensorDataClassMixin
from neurox.execution.loop import (
    run_while_loop_with_counter,
)


class _State(TensorDataClassMixin, PyTreeDataClassMixin):
    value: Tensor


def _count_down(value: Tensor, max_iter: int) -> _State:
    def cond_fn(step: Tensor, state: _State) -> Tensor:
        return (step < max_iter) & (state.value > 0).any()

    def body_fn(_step: Tensor, state: _State) -> _State:
        return _State(value=(state.value - 1).clamp_min(0))

    return run_while_loop_with_counter(
        init_state=_State(value=value),
        cond_fn=cond_fn,
        body_fn=body_fn,
        device=value.device,
    )


@pytest.mark.parametrize("max_iter", [0, 2, 5])
def test_caller_condition_bounds_updates(device: torch.device, max_iter: int) -> None:
    value = torch.tensor([0.0, 1.0, 3.0], device=device, dtype=torch.float64)
    original = value.clone()

    state = _count_down(value, max_iter)

    torch.testing.assert_close(state.value, (original - max_iter).clamp_min(0))
    torch.testing.assert_close(value, original)


def test_body_receives_zero_based_tensor_step(device: torch.device) -> None:
    value = torch.zeros((), device=device, dtype=torch.float64)

    def cond_fn(step: Tensor, _state: _State) -> Tensor:
        return step < 4

    def body_fn(step: Tensor, state: _State) -> _State:
        return _State(value=state.value + step.to(state.value.dtype))

    state = run_while_loop_with_counter(
        init_state=_State(value=value),
        cond_fn=cond_fn,
        body_fn=body_fn,
        device=device,
    )

    torch.testing.assert_close(state.value, torch.tensor(6.0, device=device, dtype=value.dtype))

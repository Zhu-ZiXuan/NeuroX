"""Input phases preserve matrix products while bounding simultaneous activation."""

from __future__ import annotations

import pytest
import torch

from neurox.architecture.mapping import InputActivation


@pytest.mark.parametrize(
    ("input_num", "max_active_num", "phase_num"),
    [(8, 8, 1), (8, 2, 4), (10, 3, 4), (3, 2, 2), (25, 16, 2)],
)
def test_input_phases_recover_matrix_products(
    input_num: int, max_active_num: int, phase_num: int, device: torch.device
) -> None:
    activation = InputActivation(input_num=input_num, max_active_num=max_active_num)
    # Shape: [vector, input]
    x = torch.arange(1, 2 * input_num + 1, device=device).reshape(2, input_num)
    # Shape: [input, output]
    w = torch.arange(input_num * 3, device=device).reshape(input_num, 3) % 7 - 3
    # Shape: [vector, input_phase, input]
    phased = activation.map_x(x)
    assert activation.input_phase_num == phase_num
    assert phased.shape == (2, phase_num, input_num)
    assert torch.all(torch.count_nonzero(phased, dim=-1) <= max_active_num)
    # Shape: [vector, input_phase, output]
    partial = (phased.unsqueeze(-1) * w).sum(dim=-2)
    actual = activation.recover(partial, dim=-2)
    torch.testing.assert_close(actual, (x.cpu() @ w.cpu()).to(device))


def test_input_phases_select_consecutive_positions(device: torch.device) -> None:
    activation = InputActivation(input_num=3, max_active_num=2)
    actual = activation.map_x(torch.tensor([2, 3, 5], device=device))
    expected = torch.tensor([[2, 3, 0], [0, 0, 5]], device=device)
    torch.testing.assert_close(actual, expected)

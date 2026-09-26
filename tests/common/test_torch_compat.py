"""Scan axis normalization and reverse traversal preserve the caller's output layouts."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.common.torch_compat import torch_scan


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_dimension_and_output_layout(dim: int, reverse: bool) -> None:
    def combine_fn(carry: Tensor, item: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        updated = carry * item["scale"] + item["values"]
        return updated, {"values": updated.clone(), "total": updated.sum()}

    values = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    scales = torch.tensor([0.25, 0.5, 0.75, 1.0]).reshape(4, 1, 1)
    actual_carry, actual_output = torch_scan(
        combine_fn,
        torch.zeros(2, 3),
        {"values": values.movedim(0, dim), "scale": scales.movedim(0, dim)},
        dim=dim,
        reverse=reverse,
        output_template={"values": torch.empty(()), "total": torch.empty(())},
    )

    carry = torch.zeros(2, 3)
    history = []
    indices = range(values.shape[0])
    for index in reversed(indices) if reverse else indices:
        carry = carry * scales[index] + values[index]
        history.append(carry)
    if reverse:
        history.reverse()
    expected_values = torch.stack(history)
    expected_totals = expected_values.sum(dim=(1, 2))

    torch.testing.assert_close(actual_carry, carry)
    torch.testing.assert_close(actual_output["values"], expected_values.movedim(0, dim))
    torch.testing.assert_close(actual_output["total"], expected_totals)

"""Profiler layout preserves caller axes while reducing physical work axes."""

import pytest
import torch

from neurox import Profiler


@pytest.mark.parametrize("rank", [0, 2, 4])
def test_reduction_preserves_the_caller_prefix(rank: int) -> None:
    # Distinct extents and values expose a reduction over the wrong axes.
    energy = torch.arange(2 * 5 * 3 * 7, dtype=torch.float64).reshape(2, 5, 3, 7)
    record = Profiler(leading_rank=rank).lay_out(qualified_name="leaf", dynamic_energy__fJ=energy, channel=None)
    expected = energy.reshape(*energy.shape[:rank], -1).sum(dim=-1)
    torch.testing.assert_close(record.dynamic_energy__fJ, expected)

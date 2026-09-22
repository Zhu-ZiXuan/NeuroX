"""Block recurrences checked against independently assembled dense systems."""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.xbar.solver._linalg import (
    boundary_inverse_block_tridiagonal_2x2,
    solve_block_tridiagonal_2x2,
)


def _uniform_case(
    block_num: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    off_diag = (-4.0e3, -7.0e3)
    generator = torch.Generator(device=device).manual_seed(seed)
    diag = torch.randn(2, 5, block_num, 2, 2, dtype=torch.float64, generator=generator, device=device) * 50.0
    diag = diag + torch.eye(2, dtype=torch.float64, device=device) * 2.0 * 8.0e3
    rhs = torch.randn(
        2,
        5,
        block_num,
        2,
        dtype=torch.float64,
        generator=generator,
        device=device,
    )
    return diag, rhs, off_diag


def _dense_matrix(
    diag: torch.Tensor,
    off_diag: tuple[float, float],
) -> torch.Tensor:
    block_num = diag.shape[-3]
    dense = torch.zeros(
        *diag.shape[:-3],
        2 * block_num,
        2 * block_num,
        dtype=diag.dtype,
        device=diag.device,
    )
    off_block = torch.diag(diag.new_tensor(off_diag))
    for k in range(block_num):
        current = slice(2 * k, 2 * (k + 1))
        dense[..., current, current] = diag[..., k, :, :]
        if k > 0:
            previous = slice(2 * (k - 1), 2 * k)
            dense[..., current, previous] = off_block
        if k + 1 < block_num:
            following = slice(2 * (k + 1), 2 * (k + 2))
            dense[..., current, following] = off_block
    return dense


@pytest.mark.parametrize(("block_num", "dim"), [(1, -1), (2, -1), (3, -1), (9, -1), (7, -2), (7, 0)])
def test_block_solution_and_boundary_inverse_match_dense_system(block_num: int, dim: int, device: torch.device) -> None:
    diag, rhs, off_diag = _uniform_case(block_num, device=device, seed=block_num * 17 + 3)
    dense = _dense_matrix(diag, off_diag)
    expected_solution = torch.linalg.solve(dense, rhs.flatten(-2).unsqueeze(-1)).squeeze(-1).view_as(rhs)
    expected_boundary = torch.linalg.inv(dense)[..., :2, :2]
    components = tuple(
        diag[..., row, col].movedim(-1, dim).contiguous() for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    rhs_components = tuple(rhs[..., component].movedim(-1, dim).contiguous() for component in (0, 1))

    solution = solve_block_tridiagonal_2x2(components, rhs_components, off_diag=off_diag, dim=dim)
    boundary = boundary_inverse_block_tridiagonal_2x2(components, off_diag=off_diag, dim=dim)
    for component, value in enumerate(solution):
        torch.testing.assert_close(value, expected_solution[..., component].movedim(-1, dim), rtol=1e-10, atol=0)
    for value, (row, col) in zip(boundary, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        torch.testing.assert_close(value, expected_boundary[..., row, col].unsqueeze(dim), rtol=1e-10, atol=0)

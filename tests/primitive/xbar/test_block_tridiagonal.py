"""Tests for the component-wise 2×2 linear-algebra specializations."""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.xbar.solver._linalg import (
    boundary_inverse_block_tridiagonal_2x2,
    solve_2x2,
    solve_block_tridiagonal_2x2,
)


def test_solve_2x2_matches_dense_reference(device: torch.device) -> None:
    """The tuple specialization agrees with a pivoted dense reference."""
    matrix = torch.randn(3, 5, 2, 2, dtype=torch.float64, device=device)
    matrix = matrix + 3.0 * torch.eye(2, dtype=torch.float64, device=device)
    rhs = torch.randn(3, 5, 2, dtype=torch.float64, device=device)

    expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    actual_0, actual_1 = solve_2x2(
        (matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 1, 0], matrix[..., 1, 1]),
        (rhs[..., 0], rhs[..., 1]),
    )

    torch.testing.assert_close(actual_0, expected[..., 0])
    torch.testing.assert_close(actual_1, expected[..., 1])


def _uniform_case(
    block_num: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    """Build a batched wire-Newton system with one constant off-block."""
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
    """Materialize one test-only dense block-tridiagonal matrix."""
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


def _dense_reference(
    diag: torch.Tensor,
    rhs: torch.Tensor,
    off_diag: tuple[float, float],
) -> torch.Tensor:
    """Solve one test-only dense reference system."""
    dense = _dense_matrix(diag, off_diag)
    flat_rhs = rhs.flatten(-2).unsqueeze(-1)
    return torch.linalg.solve(dense, flat_rhs).squeeze(-1).view_as(rhs)


@pytest.mark.parametrize("block_num", [1, 2, 3, 9])
def test_uniform_boundary_inverse_matches_dense_reference(
    block_num: int,
    device: torch.device,
) -> None:
    """The reverse Schur sweep returns the leading inverse block."""
    diag, _rhs, off_diag = _uniform_case(block_num, device=device, seed=block_num * 29 + 5)
    dense_inverse_boundary = torch.linalg.inv(_dense_matrix(diag, off_diag))[..., :2, :2]

    actual_00, actual_01, actual_10, actual_11 = boundary_inverse_block_tridiagonal_2x2(
        diag=(diag[..., 0, 0].clone(), diag[..., 0, 1].clone(), diag[..., 1, 0].clone(), diag[..., 1, 1].clone()),
        off_diag=off_diag,
        dim=-1,
    )

    torch.testing.assert_close(actual_00, dense_inverse_boundary[..., 0, 0].unsqueeze(-1))
    torch.testing.assert_close(actual_01, dense_inverse_boundary[..., 0, 1].unsqueeze(-1))
    torch.testing.assert_close(actual_10, dense_inverse_boundary[..., 1, 0].unsqueeze(-1))
    torch.testing.assert_close(actual_11, dense_inverse_boundary[..., 1, 1].unsqueeze(-1))


@pytest.mark.parametrize("block_num", [1, 2, 3, 9])
def test_uniform_block_tridiagonal_matches_dense_reference(
    block_num: int,
    device: torch.device,
) -> None:
    """The tuple recurrence solves the declared constant-off-block system."""
    diag, rhs, off_diag = _uniform_case(block_num, device=device, seed=block_num * 17 + 3)
    expected = _dense_reference(diag, rhs, off_diag)

    x_0, x_1 = solve_block_tridiagonal_2x2(
        diag=(diag[..., 0, 0].clone(), diag[..., 0, 1].clone(), diag[..., 1, 0].clone(), diag[..., 1, 1].clone()),
        rhs=(rhs[..., 0].clone(), rhs[..., 1].clone()),
        off_diag=off_diag,
        dim=-1,
    )
    actual = torch.stack((x_0, x_1), dim=-1)

    assert actual.shape == (2, 5, block_num, 2)
    relative_error = (actual - expected).abs().max() / expected.abs().max()
    assert relative_error < 1e-10, f"N={block_num}: relative error {relative_error.item():.2e}"


@pytest.mark.parametrize("block_num", [1, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("singleton_batch", [False, True])
def test_block_scans_compile_with_strided_inputs(
    block_num: int, dtype: torch.dtype, singleton_batch: bool, device: torch.device
) -> None:
    """Both sweeps preserve row order and handle a single block under compilation."""
    diag, rhs, off_diag = _uniform_case(block_num, device=device, seed=13)
    diag = diag[0, 0].to(dtype)
    rhs = rhs[0, 0].to(dtype)
    if singleton_batch:
        diag = diag[None, None]
        rhs = rhs[None, None]
    expected = _dense_reference(diag.double(), rhs.double(), off_diag).to(dtype)

    def solve(diag: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return solve_block_tridiagonal_2x2(
            diag=(diag[..., 0, 0], diag[..., 0, 1], diag[..., 1, 0], diag[..., 1, 1]),
            rhs=(rhs[..., 0], rhs[..., 1]),
            off_diag=off_diag,
            dim=-1,
        )

    with torch.no_grad():
        x_0, x_1 = torch.compile(solve, fullgraph=True, dynamic=False)(diag, rhs)
    actual = torch.stack((x_0, x_1), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=3e-5 if dtype == torch.float32 else 1e-10, atol=0)


@pytest.mark.parametrize("block_num", [1, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("singleton_batch", [False, True])
def test_boundary_scan_compiles_with_strided_inputs(
    block_num: int, dtype: torch.dtype, singleton_batch: bool, device: torch.device
) -> None:
    diag, _rhs, off_diag = _uniform_case(block_num, device=device, seed=31)
    diag = diag[0, 0].to(dtype)
    if singleton_batch:
        diag = diag.unsqueeze(0).unsqueeze(0)
    expected = torch.linalg.inv(_dense_matrix(diag.double(), off_diag))[..., :2, :2].to(dtype)

    # Separate backing tensors preserve strided rows without aliased closure inputs.
    components = (diag.clone()[..., 0, 0], diag.clone()[..., 0, 1], diag.clone()[..., 1, 0], diag.clone()[..., 1, 1])

    def solve(
        components: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return boundary_inverse_block_tridiagonal_2x2(diag=components, off_diag=off_diag, dim=-1)

    with torch.no_grad():
        actual = torch.compile(solve, fullgraph=True, dynamic=False)(components)
    for component, (row, column) in zip(actual, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        torch.testing.assert_close(
            component, expected[..., row, column].unsqueeze(-1), rtol=3e-5 if dtype == torch.float32 else 1e-10, atol=0
        )


def test_block_scans_accept_shared_coefficient_and_rhs_tensors(device: torch.device) -> None:
    """Read-only coefficients and right-hand sides may alias across components."""
    diagonal = torch.full((2, 9), 3.0, dtype=torch.float64, device=device)
    zero = torch.zeros_like(diagonal)
    rhs = torch.arange(18, dtype=torch.float64, device=device).reshape(2, 9)
    diag = torch.diag_embed(torch.stack((diagonal, diagonal), dim=-1))
    expected = _dense_reference(diag, torch.stack((rhs, rhs), dim=-1), (-0.5, -0.5))
    x_0, x_1 = solve_block_tridiagonal_2x2(
        diag=(diagonal, zero, zero, diagonal), rhs=(rhs, rhs), off_diag=(-0.5, -0.5), dim=-1
    )
    torch.testing.assert_close(torch.stack((x_0, x_1), dim=-1), expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("dim", [-3, -2, -1, 0, 1, 2])
@pytest.mark.parametrize("block_num", [1, 7])
def test_block_scans_compile_along_selected_axis(block_num: int, dim: int, device: torch.device) -> None:
    """Both recurrences accept either axis spelling and preserve the input axis order."""
    diag, rhs, off_diag = _uniform_case(block_num, device=device, seed=43)
    expected_solution = _dense_reference(diag, rhs, off_diag)
    expected_boundary = torch.linalg.inv(_dense_matrix(diag, off_diag))[..., :2, :2]
    components = tuple(
        diag[..., row, col].movedim(-1, dim).contiguous() for row, col in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    rhs_components = tuple(rhs[..., component].movedim(-1, dim).contiguous() for component in (0, 1))

    def solve(components, rhs_components):
        solution = solve_block_tridiagonal_2x2(components, rhs_components, off_diag=off_diag, dim=dim)
        boundary = boundary_inverse_block_tridiagonal_2x2(components, off_diag=off_diag, dim=dim)
        return solution, boundary

    with torch.no_grad():
        solution, boundary = torch.compile(solve, fullgraph=True, dynamic=False)(components, rhs_components)
    for component, value in enumerate(solution):
        torch.testing.assert_close(value, expected_solution[..., component].movedim(-1, dim), rtol=1e-10, atol=0)
    for value, (row, col) in zip(boundary, ((0, 0), (0, 1), (1, 0), (1, 1)), strict=True):
        torch.testing.assert_close(value, expected_boundary[..., row, col].unsqueeze(dim), rtol=1e-10, atol=0)

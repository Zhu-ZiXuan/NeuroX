"""Direct correctness tests for ``neurox.xbar.solver.primitives.solve_tridiagonal``.

Verifies the Parallel Cyclic Reduction implementation against a dense
``torch.linalg.solve`` reference across varied ``N``, dtypes, batch
shapes, and the two wire-axes used in production (``dim=-1`` for the
BL wire, ``dim=-2`` for the SL wire).
"""

from __future__ import annotations

import math

import pytest
import torch

from neurox.xbar.solver.primitives import solve_tridiagonal


def _build_dense(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor, dim: int) -> torch.Tensor:
    """Build the dense ``[..., N, N]`` tridiagonal matrix from the three diagonals.

    ``sub[..., 0]`` and ``sup[..., -1]`` are solver placeholders; we drop
    them here so the dense reference matches the *actual* linear system
    (no spurious coupling to non-existent rows).
    """
    a = sub.movedim(dim, -1)
    b = diag.movedim(dim, -1)
    c = sup.movedim(dim, -1)
    N = b.shape[-1]
    mat = torch.diag_embed(b)
    if N > 1:
        mat = mat + torch.diag_embed(a[..., 1:], offset=-1)
        mat = mat + torch.diag_embed(c[..., :-1], offset=1)
    return mat


def _random_diag_dominant(shape: tuple[int, ...], N: int, dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Make a random diagonally-dominant tridiagonal system of length ``N`` along ``dim``.

    The boundary entries ``sub[..., 0, ...]`` and ``sup[..., N-1, ...]`` are
    filled with deliberately-noisy placeholder values (+/- large magnitude)
    so that any bug that lets them leak through PCR will be flagged.
    """
    full_shape = list(shape)
    full_shape.insert(dim if dim >= 0 else dim + len(full_shape) + 1, N)
    gen = torch.Generator().manual_seed(1234)
    sub = torch.randn(full_shape, dtype=dtype, generator=gen)
    sup = torch.randn(full_shape, dtype=dtype, generator=gen)
    # diag = 1 + |sub| + |sup|  → strictly row-wise diagonally dominant.
    diag = (sub.abs() + sup.abs() + 1.0).to(dtype)
    rhs = torch.randn(full_shape, dtype=dtype, generator=gen)
    # Poison the boundary placeholder positions so the test exercises the
    # internal zero-masking path.
    if N >= 1:
        sub = sub.clone()
        sup = sup.clone()
        sub.narrow(dim, 0, 1).fill_(7.0)
        sup.narrow(dim, N - 1, 1).fill_(-11.0)
    return sub, diag, sup, rhs


@pytest.mark.parametrize("N", [1, 2, 3, 5, 16, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 4)])
def test_pcr_matches_dense_solve(N: int, dtype: torch.dtype, batch_shape: tuple[int, ...]) -> None:
    """PCR output must match a dense LU solve within a floor × N × eps."""
    sub, diag, sup, rhs = _random_diag_dominant(batch_shape, N, dim=-1, dtype=dtype)
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
    dense = _build_dense(sub, diag, sup, dim=-1)
    reference = torch.linalg.solve(dense, rhs.unsqueeze(-1)).squeeze(-1)
    eps = torch.finfo(dtype).eps
    tol = max(16.0 * N * eps, 1e-6 if dtype is torch.float32 else 1e-12)
    assert torch.allclose(x, reference, atol=tol, rtol=tol), (
        f"PCR / dense max abs diff = {(x - reference).abs().max().item():.3e}; tol = {tol:.3e}"
    )


@pytest.mark.parametrize("dim", [-1, -2])
def test_pcr_matches_dense_solve_on_production_shapes(dim: int) -> None:
    """Mirrors the production shape: ``[batch=8, col=64, row=64]`` along either wire axis."""
    N = 64
    batch = (8,)
    sub, diag, sup, rhs = _random_diag_dominant((*batch, 64), N=N, dim=dim, dtype=torch.float32)
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=dim)
    dense = _build_dense(sub, diag, sup, dim=dim)
    rhs_moved = rhs.movedim(dim, -1)
    reference = torch.linalg.solve(dense, rhs_moved.unsqueeze(-1)).squeeze(-1).movedim(-1, dim)
    max_diff = (x - reference).abs().max().item()
    assert max_diff < 1e-4, f"PCR / dense max abs diff = {max_diff:.3e}"


def test_pcr_tolerates_non_zero_boundary_placeholders() -> None:
    """``sub[..., 0]`` and ``sup[..., -1]`` must be ignored (per contract)."""
    N = 16
    gen = torch.Generator().manual_seed(7)
    sub = torch.randn(N, generator=gen)
    sup = torch.randn(N, generator=gen)
    diag = sub.abs() + sup.abs() + 1.0
    rhs = torch.randn(N, generator=gen)

    # Both variants of the boundary placeholders must give the same answer.
    sub_variant_a = sub.clone()
    sub_variant_a[0] = 42.0
    sup_variant_a = sup.clone()
    sup_variant_a[-1] = -99.0
    sub_variant_b = sub.clone()
    sub_variant_b[0] = 0.0
    sup_variant_b = sup.clone()
    sup_variant_b[-1] = 0.0

    x_a = solve_tridiagonal(sub_variant_a, diag, sup_variant_a, rhs, dim=-1)
    x_b = solve_tridiagonal(sub_variant_b, diag, sup_variant_b, rhs, dim=-1)
    assert torch.allclose(x_a, x_b, atol=1e-6), "placeholder values leaked into the solution"


def test_pcr_pow2_and_non_pow2_sizes() -> None:
    """Regression guard: the ``while k < N`` loop must handle non-power-of-two ``N``."""
    for N in (7, 9, 15, 17, 33, 65):
        sub, diag, sup, rhs = _random_diag_dominant((), N=N, dim=-1, dtype=torch.float64)
        x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
        dense = _build_dense(sub, diag, sup, dim=-1)
        reference = torch.linalg.solve(dense, rhs.unsqueeze(-1)).squeeze(-1)
        # Diagonally-dominant → tiny error even at fp64; require < N · eps · scale.
        scale = rhs.abs().max().clamp(min=1.0)
        tol = 32.0 * N * torch.finfo(torch.float64).eps * float(scale)
        max_diff = (x - reference).abs().max().item()
        assert max_diff < max(tol, 1e-10), f"N={N}: max diff = {max_diff:.3e}, tol = {tol:.3e}"


def test_pcr_preserves_dtype_and_shape() -> None:
    """Output must match ``rhs`` exactly in dtype, device, and shape."""
    shape = (2, 3, 64, 4)
    dim = -2
    sub, diag, sup, rhs = _random_diag_dominant((2, 3, 4), N=64, dim=dim, dtype=torch.float32)
    # _random_diag_dominant inserts N at `dim` on the leading shape; re-align for this test.
    assert sub.shape == shape
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=dim)
    assert x.shape == rhs.shape
    assert x.dtype == rhs.dtype
    assert x.device == rhs.device


def test_pcr_n_equals_one_fast_path() -> None:
    """``N == 1`` must return ``rhs / diag`` without entering the PCR loop."""
    sub = torch.tensor([9.0])
    diag = torch.tensor([2.5])
    sup = torch.tensor([-7.0])
    rhs = torch.tensor([5.0])
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
    assert torch.allclose(x, torch.tensor([2.0]))
    assert math.isclose(x.item(), 5.0 / 2.5, rel_tol=1e-12)

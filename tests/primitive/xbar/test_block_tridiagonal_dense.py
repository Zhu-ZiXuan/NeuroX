"""Numerical regression for :func:`solve_block_tridiagonal_dense`.

Same accuracy bar as Thomas / PCR — assembling the dense ``NB × NB``
matrix and feeding ``torch.linalg.solve`` is the most direct path; the
test pins it against the same Thomas reference + dense LU oracle the
other solvers use.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.xbar.solver import solve_block_tridiagonal
from neurox.primitive.xbar.solver.primitives import solve_block_tridiagonal_dense


def _dense_from_blocks(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    n, b, _ = diag.shape
    out = torch.zeros(n * b, n * b, dtype=diag.dtype, device=diag.device)
    for k in range(n):
        out[k * b : (k + 1) * b, k * b : (k + 1) * b] = diag[k]
        if k + 1 < n:
            out[k * b : (k + 1) * b, (k + 1) * b : (k + 2) * b] = sup[k]
        if k > 0:
            out[k * b : (k + 1) * b, (k - 1) * b : k * b] = sub[k]
    return out


def _make_diag_dominant_blocks(
    n: int, b: int, *, dtype: torch.dtype = torch.float64, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sub = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.3
    sup = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.3
    diag = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.1
    diag = diag + torch.eye(b, dtype=dtype) * 3.0
    return sub, diag, sup


@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 3, 8, 17, 64, 128])
def test_dense_matches_lu(b: int, n: int) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(n, b, seed=n * 31 + b)
    g = torch.Generator().manual_seed(n * 13 + b * 7)
    rhs = torch.randn(n, b, dtype=torch.float64, generator=g)

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    if n == 1:
        x_ref = torch.linalg.solve(diag[0], rhs[0])
    else:
        a_dense = _dense_from_blocks(sub, diag, sup)
        x_ref = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(n, b)

    rel_err = (x_ref - x).abs().max() / (x_ref.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"B={b} N={n}: rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("n", [3, 8, 17, 64, 128, 256])
def test_dense_matches_thomas_fp64(n: int) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(n, 2, seed=n)
    g = torch.Generator().manual_seed(n * 5)
    rhs = torch.randn(n, 2, dtype=torch.float64, generator=g)

    x_dense = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    x_thomas = solve_block_tridiagonal(sub, diag, sup, rhs)
    rel_err = (x_dense - x_thomas).abs().max() / (x_thomas.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"N={n}: dense vs Thomas rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("n", [3, 8, 17, 64, 128])
def test_dense_matches_thomas_fp32(n: int) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(n, 2, dtype=torch.float32, seed=n + 1)
    g = torch.Generator().manual_seed(n * 7 + 3)
    rhs = torch.randn(n, 2, dtype=torch.float32, generator=g)

    x_dense = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    x_thomas = solve_block_tridiagonal(sub, diag, sup, rhs)
    rel_err = (x_dense - x_thomas).abs().max() / (x_thomas.abs().max() + 1e-12)
    assert rel_err < 1e-4, f"N={n}: dense vs Thomas fp32 rel err {rel_err.item():.2e}"


def test_dense_batched() -> None:
    n, b = 32, 2
    batch_shape = (4, 7)
    g = torch.Generator().manual_seed(1234)
    diag = (
        torch.eye(b, dtype=torch.float64) * 3
        + torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.1
    )
    sub = torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.3
    sup = torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.3
    rhs = torch.randn(*batch_shape, n, b, dtype=torch.float64, generator=g)

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    assert x.shape == (*batch_shape, n, b)

    for idx in [(0, 0), (3, 6), (2, 4)]:
        a_dense = _dense_from_blocks(sub[idx], diag[idx], sup[idx])
        x_ref = torch.linalg.solve(a_dense, rhs[idx].reshape(-1)).reshape(n, b)
        rel_err = (x_ref - x[idx]).abs().max() / (x_ref.abs().max() + 1e-12)
        assert rel_err < 1e-10, f"batch {idx}: rel err {rel_err.item():.2e}"


def test_dense_m_matrix_wire_jacobian_2x2() -> None:
    n, b = 64, 2
    wire_g = 5.0e3
    a = 100.0
    b_cross = -50.0
    diag = torch.zeros(n, b, b, dtype=torch.float64)
    diag[..., 0, 0] = 2 * wire_g + a
    diag[..., 0, 1] = b_cross
    diag[..., 1, 0] = -a
    diag[..., 1, 1] = 2 * wire_g - b_cross
    off = torch.zeros(n, b, b, dtype=torch.float64)
    off[..., 0, 0] = -wire_g
    off[..., 1, 1] = -wire_g
    sub = off.clone()
    sup = off.clone()
    rhs = torch.randn(n, b, dtype=torch.float64, generator=torch.Generator().manual_seed(99))

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    a_dense = _dense_from_blocks(sub, diag, sup)
    x_ref = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(n, b)

    rel_err = (x - x_ref).abs().max() / (x_ref.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"M-matrix 2×2: rel err {rel_err.item():.2e}"

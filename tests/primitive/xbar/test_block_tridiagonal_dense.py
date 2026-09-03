"""Numerical regression for `solve_block_tridiagonal_dense`.

Same accuracy bar as Thomas / PCR — assembling the dense `N*B × N*B`
matrix and feeding `torch.linalg.solve` is the most direct path; the
test pins it against the same Thomas reference + dense LU oracle the
other solvers use.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.xbar.solver._linalg import solve_block_tridiagonal, solve_block_tridiagonal_dense


def _dense_from_blocks(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    block_num, block_size, _ = diag.shape
    out = torch.zeros(block_num * block_size, block_num * block_size, dtype=diag.dtype, device=diag.device)
    for k in range(block_num):
        out[k * block_size : (k + 1) * block_size, k * block_size : (k + 1) * block_size] = diag[k]
        if k + 1 < block_num:
            out[k * block_size : (k + 1) * block_size, (k + 1) * block_size : (k + 2) * block_size] = sup[k]
        if k > 0:
            out[k * block_size : (k + 1) * block_size, (k - 1) * block_size : k * block_size] = sub[k]
    return out


def _make_diag_dominant_blocks(
    block_num: int, block_size: int, *, dtype: torch.dtype = torch.float64, seed: int = 0, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    sub = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.3
    sup = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.3
    diag = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.1
    diag = diag + torch.eye(block_size, dtype=dtype, device=device) * 3.0
    return sub, diag, sup


@pytest.mark.parametrize("block_size", [2, 3, 4])
@pytest.mark.parametrize("block_num", [1, 2, 3, 8, 17, 64, 128])
def test_dense_matches_lu(block_size: int, block_num: int, device: torch.device) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(block_num, block_size, seed=block_num * 31 + block_size, device=device)
    g = torch.Generator(device=device).manual_seed(block_num * 13 + block_size * 7)
    rhs = torch.randn(block_num, block_size, dtype=torch.float64, generator=g, device=device)

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    if block_num == 1:
        x_ref = torch.linalg.solve(diag[0], rhs[0])
    else:
        a_dense = _dense_from_blocks(sub, diag, sup)
        x_ref = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(block_num, block_size)

    # The block-row axis survives however short it is, `block_num = 1` included.
    assert x.shape == (block_num, block_size)
    rel_err = (x_ref - x).abs().max() / (x_ref.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"B={block_size} N={block_num}: rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("block_num", [3, 8, 17, 64, 128, 256])
def test_dense_matches_thomas_fp64(block_num: int, device: torch.device) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(block_num, 2, seed=block_num, device=device)
    g = torch.Generator(device=device).manual_seed(block_num * 5)
    rhs = torch.randn(block_num, 2, dtype=torch.float64, generator=g, device=device)

    x_dense = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    x_thomas = solve_block_tridiagonal(sub, diag, sup, rhs)
    rel_err = (x_dense - x_thomas).abs().max() / (x_thomas.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"N={block_num}: dense vs Thomas rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("block_num", [3, 8, 17, 64, 128])
def test_dense_matches_thomas_fp32(block_num: int, device: torch.device) -> None:
    sub, diag, sup = _make_diag_dominant_blocks(block_num, 2, dtype=torch.float32, seed=block_num + 1, device=device)
    g = torch.Generator(device=device).manual_seed(block_num * 7 + 3)
    rhs = torch.randn(block_num, 2, dtype=torch.float32, generator=g, device=device)

    x_dense = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    x_thomas = solve_block_tridiagonal(sub, diag, sup, rhs)
    rel_err = (x_dense - x_thomas).abs().max() / (x_thomas.abs().max() + 1e-12)
    assert rel_err < 1e-4, f"N={block_num}: dense vs Thomas fp32 rel err {rel_err.item():.2e}"


def test_dense_batched(device: torch.device) -> None:
    block_num = 32
    block_size = 2
    batch_shape = (4, 7)
    g = torch.Generator(device=device).manual_seed(1234)
    diag = (
        torch.eye(block_size, dtype=torch.float64, device=device) * 3
        + torch.randn(*batch_shape, block_num, block_size, block_size, dtype=torch.float64, generator=g, device=device)
        * 0.1
    )
    sub = (
        torch.randn(*batch_shape, block_num, block_size, block_size, dtype=torch.float64, generator=g, device=device)
        * 0.3
    )
    sup = (
        torch.randn(*batch_shape, block_num, block_size, block_size, dtype=torch.float64, generator=g, device=device)
        * 0.3
    )
    rhs = torch.randn(*batch_shape, block_num, block_size, dtype=torch.float64, generator=g, device=device)

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    assert x.shape == (*batch_shape, block_num, block_size)

    for idx in [(0, 0), (3, 6), (2, 4)]:
        a_dense = _dense_from_blocks(sub[idx], diag[idx], sup[idx])
        x_ref = torch.linalg.solve(a_dense, rhs[idx].reshape(-1)).reshape(block_num, block_size)
        rel_err = (x_ref - x[idx]).abs().max() / (x_ref.abs().max() + 1e-12)
        assert rel_err < 1e-10, f"batch {idx}: rel err {rel_err.item():.2e}"


def test_dense_m_matrix_wire_jacobian_2x2(device: torch.device) -> None:
    block_num = 64
    block_size = 2
    wire_g = 5.0e3
    a = 100.0
    b_cross = -50.0
    diag = torch.zeros(block_num, block_size, block_size, dtype=torch.float64, device=device)
    diag[..., 0, 0] = 2 * wire_g + a
    diag[..., 0, 1] = b_cross
    diag[..., 1, 0] = -a
    diag[..., 1, 1] = 2 * wire_g - b_cross
    off = torch.zeros(block_num, block_size, block_size, dtype=torch.float64, device=device)
    off[..., 0, 0] = -wire_g
    off[..., 1, 1] = -wire_g
    sub = off.clone()
    sup = off.clone()
    rhs = torch.randn(
        block_num,
        block_size,
        dtype=torch.float64,
        generator=torch.Generator(device=device).manual_seed(99),
        device=device,
    )

    x = solve_block_tridiagonal_dense(sub, diag, sup, rhs)
    a_dense = _dense_from_blocks(sub, diag, sup)
    x_ref = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(block_num, block_size)

    rel_err = (x - x_ref).abs().max() / (x_ref.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"M-matrix 2×2: rel err {rel_err.item():.2e}"

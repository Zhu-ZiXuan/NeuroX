"""Unit tests for :func:`neurox.xbar.solver.solve_block_tridiagonal`.

Covers:
  * ``block_size = 1`` reduces to the existing scalar Thomas solver
    (`solve_tridiagonal`) bit-exact.
  * ``block_size ∈ {2, 3}`` matches a dense reference solve via
    ``torch.linalg.solve`` on the equivalent dense matrix.
  * Batch dims pass through correctly.
  * Numerically stable on diagonally-dominant (M-matrix-flavour) systems
    that mirror the wire-Newton + boundary block structure used by the
    nested solver.
"""

from __future__ import annotations

import pytest
import torch

from neurox.xbar.solver import solve_block_tridiagonal
from neurox.xbar.solver.primitives import solve_tridiagonal


def _dense_from_blocks(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """Materialize the dense ``N*B × N*B`` matrix from block tridiagonal data.

    ``sub[0]`` and ``sup[-1]`` are unused placeholders by convention.
    """
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
    """Random block-tridiagonal where each diag block is diagonally dominant."""
    g = torch.Generator().manual_seed(seed)
    sub = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.3
    sup = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.3
    diag = torch.randn(n, b, b, dtype=dtype, generator=g) * 0.1
    # Add dominant diagonal to every block.
    diag = diag + torch.eye(b, dtype=dtype) * 3.0
    return sub, diag, sup


def test_block_size_1_matches_scalar_thomas() -> None:
    """B = 1 should agree with scalar Thomas to fp64 round-off.

    Not bit-exact — block path goes through ``torch.linalg.solve`` (LU)
    while scalar Thomas does explicit division. Same answer modulo
    accumulation order. ~1e-14 relative tolerance is appropriate.
    """
    n = 12
    g = torch.Generator().manual_seed(42)
    diag_s = torch.rand(n, dtype=torch.float64, generator=g) + 1.0
    sub_s = torch.rand(n, dtype=torch.float64, generator=g) * 0.1
    sup_s = torch.rand(n, dtype=torch.float64, generator=g) * 0.1
    rhs_s = torch.rand(n, dtype=torch.float64, generator=g)

    x_scalar = solve_tridiagonal(sub_s, diag_s, sup_s, rhs_s, dim=0)
    x_block = solve_block_tridiagonal(
        sub_s.unsqueeze(-1).unsqueeze(-1),
        diag_s.unsqueeze(-1).unsqueeze(-1),
        sup_s.unsqueeze(-1).unsqueeze(-1),
        rhs_s.unsqueeze(-1),
    ).squeeze(-1)

    rel_err = (x_scalar - x_block).abs().max() / x_scalar.abs().max()
    assert rel_err < 1e-13, f"rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("b", [2, 3, 4])
@pytest.mark.parametrize("n", [1, 3, 8, 17])
def test_block_solve_matches_dense(b: int, n: int) -> None:
    """Block Thomas matches dense ``torch.linalg.solve`` on the assembled matrix."""
    sub, diag, sup = _make_diag_dominant_blocks(n, b, seed=n * 31 + b)
    g = torch.Generator().manual_seed(n * 13 + b * 7)
    rhs = torch.randn(n, b, dtype=torch.float64, generator=g)

    if n == 1:
        x_dense = torch.linalg.solve(diag[0], rhs[0])
    else:
        a_dense = _dense_from_blocks(sub, diag, sup)
        x_dense = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(n, b)
    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)

    rel_err = (x_dense - x_block).abs().max() / (x_dense.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"B={b} N={n}: rel error {rel_err.item():.2e}"


def test_batched_block_solve() -> None:
    """Leading batch dims pass through; per-batch result matches dense."""
    n, b = 8, 3
    batch_shape = (4, 7)
    g = torch.Generator().manual_seed(1234)
    diag = (
        torch.eye(b, dtype=torch.float64) * 3
        + torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.1
    )
    sub = torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.3
    sup = torch.randn(*batch_shape, n, b, b, dtype=torch.float64, generator=g) * 0.3
    rhs = torch.randn(*batch_shape, n, b, dtype=torch.float64, generator=g)

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    assert x_block.shape == (*batch_shape, n, b)

    # Spot-check a few batch elements against dense.
    for idx in [(0, 0), (3, 6), (2, 4)]:
        a_dense = _dense_from_blocks(sub[idx], diag[idx], sup[idx])
        x_dense = torch.linalg.solve(a_dense, rhs[idx].reshape(-1)).reshape(n, b)
        rel_err = (x_dense - x_block[idx]).abs().max() / (x_dense.abs().max() + 1e-12)
        assert rel_err < 1e-10, f"batch {idx}: rel error {rel_err.item():.2e}"


def test_zero_off_diagonals_reduce_to_block_diag() -> None:
    """Sub/sup all zero → solution is per-block independent solve."""
    n, b = 5, 2
    g = torch.Generator().manual_seed(7)
    diag = torch.eye(b, dtype=torch.float64) * 2 + torch.randn(n, b, b, dtype=torch.float64, generator=g) * 0.05
    sub = torch.zeros(n, b, b, dtype=torch.float64)
    sup = torch.zeros(n, b, b, dtype=torch.float64)
    rhs = torch.randn(n, b, dtype=torch.float64, generator=g)

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    for k in range(n):
        x_expected = torch.linalg.solve(diag[k], rhs[k])
        assert torch.allclose(x_block[k], x_expected, atol=1e-12)


def test_m_matrix_block_2x2_mirrors_nested_wire_jacobian() -> None:
    """Reproduce the BL/SL block-2×2 wire-Newton structure used by the
    Phase-C nested solver upgrade: positive diagonal blocks, scalar
    negative off-diagonals (wire coupling), small off-diagonal cell
    cross-terms. Must solve cleanly even when off-diagonals are present.
    """
    n = 16
    b = 2
    wire_g = 5.0e3  # ~ our chip's bl_segment_g[1:] scale (uS)
    a = 100.0  # ∂I_cell/∂V_BL ≈ g_R · g_ND / D
    b_cross = -50.0  # ∂I_cell/∂V_SL (negative)

    # Diagonal block: [[wire_diag + a, b_cross], [-a, wire_diag - b_cross]].
    diag = torch.zeros(n, b, b, dtype=torch.float64)
    diag[..., 0, 0] = 2 * wire_g + a
    diag[..., 0, 1] = b_cross
    diag[..., 1, 0] = -a
    diag[..., 1, 1] = 2 * wire_g - b_cross
    # Off-diagonal blocks: diagonal 2×2 with -wire_g on BL-BL and SL-SL only.
    off = torch.zeros(n, b, b, dtype=torch.float64)
    off[..., 0, 0] = -wire_g
    off[..., 1, 1] = -wire_g
    sub = off.clone()
    sup = off.clone()

    rhs = torch.randn(n, b, dtype=torch.float64, generator=torch.Generator().manual_seed(99))

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    a_dense = _dense_from_blocks(sub, diag, sup)
    x_dense = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(n, b)

    rel_err = (x_dense - x_block).abs().max() / (x_dense.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"M-matrix-like 2×2 block: rel err {rel_err.item():.2e}"

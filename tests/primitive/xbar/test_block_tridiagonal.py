"""Unit tests for the block-tridiagonal solvers in
`neurox.primitive.xbar.solver._linalg`.

Covers, for the general `solve_block_tridiagonal`:
  * `block_size = 1` reduces to the existing scalar Thomas solver
    (`solve_tridiagonal`) bit-exact.
  * `block_size` of 2 or 3 matches a dense reference solve via
    `torch.linalg.solve` on the equivalent dense matrix.
  * Batch dims pass through correctly.
  * Numerically stable on diagonally-dominant (M-matrix-flavour) systems
    that mirror the wire-Newton + boundary block structure used by the
    nested solver.

And, for the specialized `solve_block_tridiagonal_2x2_uniform`:
  * EQUIVALENCE LAW: it solves the very system the general kernel solves
    when handed that system's constant off-block materialized.
  * NO-BOUNDARY LAW: a constant off-block needs no boundary slots, so the
    answer cannot depend on what the general kernel would have found in
    the two unused ones.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.xbar.solver._linalg import (
    solve_block_tridiagonal,
    solve_block_tridiagonal_2x2_uniform,
    solve_tridiagonal,
)


def _dense_from_blocks(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
    """Materialize the dense `N*B × N*B` matrix from block tridiagonal data.

    `sub[0]` and `sup[-1]` are unused placeholders by convention.
    """
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
    """Random block-tridiagonal where each diag block is diagonally dominant."""
    g = torch.Generator(device=device).manual_seed(seed)
    sub = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.3
    sup = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.3
    diag = torch.randn(block_num, block_size, block_size, dtype=dtype, generator=g, device=device) * 0.1
    diag = diag + torch.eye(block_size, dtype=dtype, device=device) * 3.0
    return sub, diag, sup


def test_block_size_1_matches_scalar_thomas(device: torch.device) -> None:
    """B = 1 should agree with scalar Thomas to fp64 round-off.

    Not bit-exact — block path goes through `torch.linalg.solve` (LU)
    while scalar Thomas does explicit division. Same answer modulo
    accumulation order. ~1e-14 relative tolerance is appropriate.
    """
    block_num = 12
    g = torch.Generator(device=device).manual_seed(42)
    diag_s = torch.rand(block_num, dtype=torch.float64, generator=g, device=device) + 1.0
    sub_s = torch.rand(block_num, dtype=torch.float64, generator=g, device=device) * 0.1
    sup_s = torch.rand(block_num, dtype=torch.float64, generator=g, device=device) * 0.1
    rhs_s = torch.rand(block_num, dtype=torch.float64, generator=g, device=device)

    x_scalar = solve_tridiagonal(sub_s, diag_s, sup_s, rhs_s, dim=0)
    x_block = solve_block_tridiagonal(
        sub_s.view(block_num, 1, 1),
        diag_s.view(block_num, 1, 1),
        sup_s.view(block_num, 1, 1),
        rhs_s.unsqueeze(-1),
    ).squeeze(-1)

    rel_err = (x_scalar - x_block).abs().max() / x_scalar.abs().max()
    assert rel_err < 1e-13, f"rel err {rel_err.item():.2e}"


@pytest.mark.parametrize("block_size", [2, 3, 4])
@pytest.mark.parametrize("block_num", [1, 3, 8, 17])
def test_block_solve_matches_dense(block_size: int, block_num: int, device: torch.device) -> None:
    """Block Thomas matches dense `torch.linalg.solve` on the assembled matrix."""
    sub, diag, sup = _make_diag_dominant_blocks(block_num, block_size, seed=block_num * 31 + block_size, device=device)
    g = torch.Generator(device=device).manual_seed(block_num * 13 + block_size * 7)
    rhs = torch.randn(block_num, block_size, dtype=torch.float64, generator=g, device=device)

    if block_num == 1:
        x_dense = torch.linalg.solve(diag[0], rhs[0])
    else:
        a_dense = _dense_from_blocks(sub, diag, sup)
        x_dense = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(block_num, block_size)
    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)

    # The block-row axis survives however short it is, `block_num = 1` included.
    assert x_block.shape == (block_num, block_size)
    rel_err = (x_dense - x_block).abs().max() / (x_dense.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"B={block_size} N={block_num}: rel error {rel_err.item():.2e}"


def test_batched_block_solve(device: torch.device) -> None:
    """Leading batch dims pass through; per-batch result matches dense."""
    block_num = 8
    block_size = 3
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

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    assert x_block.shape == (*batch_shape, block_num, block_size)

    # Spot-check a few batch elements against dense.
    for idx in [(0, 0), (3, 6), (2, 4)]:
        a_dense = _dense_from_blocks(sub[idx], diag[idx], sup[idx])
        x_dense = torch.linalg.solve(a_dense, rhs[idx].reshape(-1)).reshape(block_num, block_size)
        rel_err = (x_dense - x_block[idx]).abs().max() / (x_dense.abs().max() + 1e-12)
        assert rel_err < 1e-10, f"batch {idx}: rel error {rel_err.item():.2e}"


def test_zero_off_diagonals_reduce_to_block_diag(device: torch.device) -> None:
    """Sub/sup all zero → solution is per-block independent solve."""
    block_num = 5
    block_size = 2
    g = torch.Generator(device=device).manual_seed(7)
    diag = (
        torch.eye(block_size, dtype=torch.float64, device=device) * 2
        + torch.randn(block_num, block_size, block_size, dtype=torch.float64, generator=g, device=device) * 0.05
    )
    sub = torch.zeros(block_num, block_size, block_size, dtype=torch.float64, device=device)
    sup = torch.zeros(block_num, block_size, block_size, dtype=torch.float64, device=device)
    rhs = torch.randn(block_num, block_size, dtype=torch.float64, generator=g, device=device)

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    for k in range(block_num):
        x_expected = torch.linalg.solve(diag[k], rhs[k])
        assert torch.allclose(x_block[k], x_expected, atol=1e-12)


def test_m_matrix_block_2x2_mirrors_nested_wire_jacobian(device: torch.device) -> None:
    """Reproduce the BL/SL block-2×2 wire-Newton structure used by the
    nested solver: positive diagonal blocks, scalar negative off-diagonals
    (wire coupling), small off-diagonal cell cross-terms. Must solve
    cleanly even when off-diagonals are present.
    """
    block_num = 16
    block_size = 2
    wire_g = 5.0e3  # ~ a chip's per-link rail conductance scale (uS)
    a = 100.0  # ∂I_cell/∂V_BL, about g_R · g_ND / D
    b_cross = -50.0  # ∂I_cell/∂V_SL (negative)

    # Diagonal block: [[wire_diag + a, b_cross], [-a, wire_diag - b_cross]].
    diag = torch.zeros(block_num, block_size, block_size, dtype=torch.float64, device=device)
    diag[..., 0, 0] = 2 * wire_g + a
    diag[..., 0, 1] = b_cross
    diag[..., 1, 0] = -a
    diag[..., 1, 1] = 2 * wire_g - b_cross
    # Off-diagonal blocks: diagonal 2×2 with -wire_g on BL-BL and SL-SL only.
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

    x_block = solve_block_tridiagonal(sub, diag, sup, rhs)
    a_dense = _dense_from_blocks(sub, diag, sup)
    x_dense = torch.linalg.solve(a_dense, rhs.reshape(-1)).reshape(block_num, block_size)

    rel_err = (x_dense - x_block).abs().max() / (x_dense.abs().max() + 1e-12)
    assert rel_err < 1e-10, f"M-matrix-like 2×2 block: rel err {rel_err.item():.2e}"


def _uniform_case(
    block_num: int, *, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
    """A batched wire-Newton-flavoured system with one constant off-block."""
    off = (-4.0e3, -7.0e3)
    g = torch.Generator(device=device).manual_seed(seed)
    diag = torch.randn(2, 5, block_num, 2, 2, dtype=torch.float64, generator=g, device=device) * 50.0
    diag = diag + torch.eye(2, dtype=torch.float64, device=device) * 2.0 * 8.0e3
    rhs = torch.randn(2, 5, block_num, 2, dtype=torch.float64, generator=g, device=device)
    return diag, rhs, off


@pytest.mark.parametrize("block_num", [1, 2, 3, 9])
def test_uniform_off_block_matches_the_general_kernel(block_num: int, device: torch.device) -> None:
    """Same system, same answer: the specialization only skips work."""
    diag, rhs, off = _uniform_case(block_num, device=device, seed=block_num * 17 + 3)
    off_blocks = torch.diag(torch.tensor(off, dtype=torch.float64, device=device)).expand_as(diag)

    expected = solve_block_tridiagonal(off_blocks, diag, off_blocks, rhs)
    got = solve_block_tridiagonal_2x2_uniform(diag, rhs, off_block=off)

    # The block-row axis survives however short it is, `block_num = 1` included.
    assert got.shape == (2, 5, block_num, 2)
    rel_err = (got - expected).abs().max() / expected.abs().max()
    assert rel_err < 1e-10, f"N={block_num}: rel error {rel_err.item():.2e}"


def test_uniform_off_block_has_no_boundary_slots(device: torch.device) -> None:
    """One constant off-block means no boundary entry exists to get wrong.

    The general kernel keeps a sub-block at row 0 and a super-block at the
    last row that the recurrence never reads. Feeding it nonsense there and
    still landing on the specialization's answer is what says the scalar
    form carries no hidden boundary convention.
    """
    diag, rhs, off = _uniform_case(7, device=device, seed=404)
    off_blocks = torch.diag(torch.tensor(off, dtype=torch.float64, device=device)).expand_as(diag)
    sub = off_blocks.clone()
    sub[..., 0, :, :] = 1.0e6
    sup = off_blocks.clone()
    sup[..., -1, :, :] = -2.0e6

    expected = solve_block_tridiagonal(sub, diag, sup, rhs)
    got = solve_block_tridiagonal_2x2_uniform(diag, rhs, off_block=off)

    rel_err = (got - expected).abs().max() / expected.abs().max()
    assert rel_err < 1e-10, f"rel error {rel_err.item():.2e}"

"""Direct correctness tests for `neurox.primitive.xbar.solver._linalg.solve_tridiagonal`.

Verifies the Thomas sweep against a dense
`torch.linalg.solve` reference across varied `block_num`, dtypes, batch
shapes, and the two wire-axes used in production (`dim=-1` for the
BL wire, `dim=-2` for the SL wire).
"""

from __future__ import annotations

import math

import pytest
import torch

from neurox.primitive.xbar.solver._linalg import solve_tridiagonal


def _build_dense(sub: torch.Tensor, diag: torch.Tensor, sup: torch.Tensor, dim: int) -> torch.Tensor:
    """Build the dense tridiagonal matrix from the three diagonals.

    `sub[..., 0]` and `sup[..., -1]` are solver placeholders, dropped
    here so the dense reference matches the actual linear system
    (no spurious coupling to non-existent rows).
    """
    a = sub.movedim(dim, -1)
    b = diag.movedim(dim, -1)
    c = sup.movedim(dim, -1)
    block_num = b.shape[-1]
    mat = torch.diag_embed(b)
    if block_num > 1:
        mat = mat + torch.diag_embed(a[..., 1:], offset=-1)
        mat = mat + torch.diag_embed(c[..., :-1], offset=1)
    return mat


def _random_diag_dominant(
    shape: tuple[int, ...], block_num: int, dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Make a random diagonally-dominant tridiagonal system of length `block_num` along `dim`.

    The boundary entries `sub[..., 0, ...]` and `sup[..., block_num-1, ...]` are
    filled with deliberately-noisy placeholder values (± large magnitude)
    so that any bug that lets them leak into the sweep will be flagged.
    """
    full_shape = list(shape)
    full_shape.insert(dim if dim >= 0 else dim + len(full_shape) + 1, block_num)
    gen = torch.Generator(device=device).manual_seed(1234)
    sub = torch.randn(full_shape, dtype=dtype, generator=gen, device=device)
    sup = torch.randn(full_shape, dtype=dtype, generator=gen, device=device)
    # diag = 1 + |sub| + |sup|  → strictly row-wise diagonally dominant.
    diag = (sub.abs() + sup.abs() + 1.0).to(dtype)
    rhs = torch.randn(full_shape, dtype=dtype, generator=gen, device=device)
    # Poison the boundary placeholder positions so the test exercises the
    # internal zero-masking path.
    if block_num >= 1:
        sub = sub.clone()
        sup = sup.clone()
        sub.narrow(dim, 0, 1).fill_(7.0)
        sup.narrow(dim, block_num - 1, 1).fill_(-11.0)
    return sub, diag, sup, rhs


@pytest.mark.parametrize("block_num", [1, 2, 3, 5, 16, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_shape", [(), (3,), (2, 4)])
def test_thomas_matches_dense_solve(
    block_num: int, dtype: torch.dtype, batch_shape: tuple[int, ...], device: torch.device
) -> None:
    """Thomas output must match a dense LU solve within a floor × block_num × eps."""
    sub, diag, sup, rhs = _random_diag_dominant(batch_shape, block_num, dim=-1, dtype=dtype, device=device)
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
    dense = _build_dense(sub, diag, sup, dim=-1)
    reference = torch.linalg.solve(dense, rhs.unsqueeze(-1)).squeeze(-1)
    eps = torch.finfo(dtype).eps
    tol = max(16.0 * block_num * eps, 1e-6 if dtype is torch.float32 else 1e-12)
    assert torch.allclose(x, reference, atol=tol, rtol=tol), (
        f"Thomas / dense max abs diff = {(x - reference).abs().max().item():.3e}; tol = {tol:.3e}"
    )


@pytest.mark.parametrize("dim", [-1, -2])
def test_thomas_matches_dense_solve_on_production_shapes(dim: int, device: torch.device) -> None:
    """Mirrors the production layout along either wire axis."""
    block_num = 64
    batch = (8,)
    sub, diag, sup, rhs = _random_diag_dominant(
        (*batch, 64), block_num=block_num, dim=dim, dtype=torch.float32, device=device
    )
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=dim)
    dense = _build_dense(sub, diag, sup, dim=dim)
    rhs_moved = rhs.movedim(dim, -1)
    reference = torch.linalg.solve(dense, rhs_moved.unsqueeze(-1)).squeeze(-1).movedim(-1, dim)
    max_diff = (x - reference).abs().max().item()
    assert max_diff < 1e-4, f"Thomas / dense max abs diff = {max_diff:.3e}"


def test_boundary_placeholders_are_ignored(device: torch.device) -> None:
    """`sub[..., 0]` and `sup[..., -1]` must be ignored (per contract)."""
    block_num = 16
    gen = torch.Generator(device=device).manual_seed(7)
    sub = torch.randn(block_num, generator=gen, device=device)
    sup = torch.randn(block_num, generator=gen, device=device)
    diag = sub.abs() + sup.abs() + 1.0
    rhs = torch.randn(block_num, generator=gen, device=device)

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


def test_matches_dense_solve_at_pow2_and_non_pow2_sizes(device: torch.device) -> None:
    """Regression guard: the sweep must handle non-power-of-two `block_num`."""
    for block_num in (7, 9, 15, 17, 33, 65):
        sub, diag, sup, rhs = _random_diag_dominant((), block_num=block_num, dim=-1, dtype=torch.float64, device=device)
        x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
        dense = _build_dense(sub, diag, sup, dim=-1)
        reference = torch.linalg.solve(dense, rhs.unsqueeze(-1)).squeeze(-1)
        # Diagonally-dominant → tiny error even at fp64; require < block_num · eps · scale.
        scale = rhs.abs().max().clamp(min=1.0)
        tol = 32.0 * block_num * torch.finfo(torch.float64).eps * float(scale)
        max_diff = (x - reference).abs().max().item()
        assert max_diff < max(tol, 1e-10), f"block_num={block_num}: max diff = {max_diff:.3e}, tol = {tol:.3e}"


def test_output_preserves_dtype_and_shape(device: torch.device) -> None:
    """Output must match `rhs` exactly in dtype, device, and shape."""
    shape = (2, 3, 64, 4)
    dim = -2
    sub, diag, sup, rhs = _random_diag_dominant((2, 3, 4), block_num=64, dim=dim, dtype=torch.float32, device=device)
    # _random_diag_dominant inserts `block_num` at `dim` on the leading shape; re-align for this test.
    assert sub.shape == shape
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=dim)
    assert x.shape == rhs.shape
    assert x.dtype == rhs.dtype
    assert x.device == rhs.device


def test_block_num_equals_one_fast_path(device: torch.device) -> None:
    """`block_num == 1` must return `rhs / diag` without entering the sweep."""
    sub = torch.tensor([9.0], device=device)
    diag = torch.tensor([2.5], device=device)
    sup = torch.tensor([-7.0], device=device)
    rhs = torch.tensor([5.0], device=device)
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=-1)
    assert torch.allclose(x, torch.tensor([2.0], device=device))
    assert math.isclose(x.item(), 5.0 / 2.5, rel_tol=1e-12)

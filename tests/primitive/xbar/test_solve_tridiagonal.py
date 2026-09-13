"""Thomas recurrence checked against independently assembled dense systems."""

from __future__ import annotations

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


@pytest.mark.parametrize(
    ("block_num", "dtype", "batch_shape", "dim"),
    [
        pytest.param(1, torch.float64, (), -1, id="single-node"),
        pytest.param(2, torch.float64, (), -1, id="two-boundaries"),
        pytest.param(7, torch.float64, (2, 3), -2, id="interior-nodes"),
        pytest.param(64, torch.float32, (2,), -1, id="long-recurrence"),
    ],
)
def test_thomas_matches_dense_solve(
    block_num: int, dtype: torch.dtype, batch_shape: tuple[int, ...], dim: int, device: torch.device
) -> None:
    """Thomas output must match a dense LU solve within a floor × block_num × eps."""
    sub, diag, sup, rhs = _random_diag_dominant(batch_shape, block_num, dim=dim, dtype=dtype, device=device)
    x = solve_tridiagonal(sub, diag, sup, rhs, dim=dim)
    dense = _build_dense(sub, diag, sup, dim=dim)
    reference = torch.linalg.solve(dense, rhs.movedim(dim, -1).unsqueeze(-1)).squeeze(-1).movedim(-1, dim)
    eps = torch.finfo(dtype).eps
    tol = max(16.0 * block_num * eps, 1e-6 if dtype is torch.float32 else 1e-12)
    torch.testing.assert_close(x, reference, atol=tol, rtol=tol)

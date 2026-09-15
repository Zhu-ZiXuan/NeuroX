"""Placement chunking and padding."""

from __future__ import annotations

import torch

from neurox.architecture.mapping.tiling import _chunk_pad_along


def test_chunk_pad_along_without_padding() -> None:
    x = torch.arange(16)
    y = _chunk_pad_along(x, dim=0, chunk_size=4, pad_value=0)
    assert y.shape == (4, 4)
    assert torch.equal(y.flatten(), x)


def test_chunk_pad_along_with_padding() -> None:
    x = torch.arange(13)
    y = _chunk_pad_along(x, dim=0, chunk_size=16, pad_value=0)
    assert y.shape == (1, 16)
    assert torch.equal(y[0, :13], x)
    assert torch.equal(y[0, 13:], torch.zeros(3, dtype=x.dtype))


def test_chunk_pad_along_accepts_negative_axis() -> None:
    x = torch.arange(60).reshape(3, 4, 5)
    y = _chunk_pad_along(x, dim=-1, chunk_size=3, pad_value=0)
    assert y.shape == (3, 4, 2, 3)

"""Snapshot shape operations preserve per-field tensor view semantics."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import SnapBase


class _DeviceSnap(SnapBase):
    g__uS: Tensor


class _CellSnap(SnapBase):
    v_wl__V: Tensor
    device: _DeviceSnap
    label: str


# --- expand ---


def test_expand_broadcasts_every_field_to_the_target_shape_without_materializing() -> None:
    snap = _CellSnap(v_wl__V=torch.tensor([1.0, 2.0]), device=_DeviceSnap(g__uS=torch.tensor([5.0])), label="row")

    expanded = snap.expand((3, 2))

    assert expanded.v_wl__V.shape == (3, 2)
    assert expanded.device.g__uS.shape == (3, 2)
    # A genuine broadcast view, not a materialized copy.
    assert expanded.v_wl__V.stride(0) == 0
    torch.testing.assert_close(expanded.v_wl__V[0], snap.v_wl__V)
    torch.testing.assert_close(expanded.v_wl__V[1], snap.v_wl__V)


# --- flatten_axes ---


def test_flatten_axes_merges_the_named_axes_of_every_field() -> None:
    snap = _CellSnap(v_wl__V=torch.randn(2, 3, 4), device=_DeviceSnap(g__uS=torch.randn(2, 3, 5)), label="row")

    flat = snap.flatten_axes(0, 1)

    assert flat.v_wl__V.shape == (6, 4)
    assert flat.device.g__uS.shape == (6, 5)
    torch.testing.assert_close(flat.v_wl__V, snap.v_wl__V.flatten(0, 1))


def test_flatten_axes_accepts_negative_dims() -> None:
    snap = _CellSnap(v_wl__V=torch.randn(2, 3, 4), device=_DeviceSnap(g__uS=torch.randn(2, 3, 4)), label="row")

    flat = snap.flatten_axes(-2, -1)

    assert flat.v_wl__V.shape == (2, 12)
    assert flat.device.g__uS.shape == (2, 12)


# --- index_select ---


def test_index_select_gathers_the_named_axis_of_every_field() -> None:
    grid = torch.arange(6.0).reshape(2, 3)
    snap = _CellSnap(v_wl__V=grid, device=_DeviceSnap(g__uS=grid), label="row")
    index = torch.tensor([0, 2])

    selected = snap.index_select(1, index)

    expected = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
    torch.testing.assert_close(selected.v_wl__V, expected)
    torch.testing.assert_close(selected.device.g__uS, expected)


def test_index_select_accepts_a_negative_dim() -> None:
    grid = torch.arange(6.0).reshape(2, 3)
    snap = _CellSnap(v_wl__V=grid, device=_DeviceSnap(g__uS=grid), label="row")

    selected = snap.index_select(-1, torch.tensor([0, 2]))

    torch.testing.assert_close(selected.v_wl__V, torch.tensor([[0.0, 2.0], [3.0, 5.0]]))


def test_nested_snapshots_are_registered_pytrees() -> None:
    snap = _CellSnap(
        v_wl__V=torch.tensor([1.0, 2.0]),
        device=_DeviceSnap(g__uS=torch.tensor([3.0, 4.0])),
        label="row",
    )
    leaves, spec = torch.utils._pytree.tree_flatten(snap)
    assert sum(isinstance(leaf, Tensor) for leaf in leaves) == 2
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)
    assert type(restored) is _CellSnap
    assert type(restored.device) is _DeviceSnap
    assert restored.v_wl__V is snap.v_wl__V
    assert restored.device.g__uS is snap.device.g__uS
    assert restored.label == snap.label

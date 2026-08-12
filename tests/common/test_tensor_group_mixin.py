"""Tests for the ``TensorGroupMixin`` per-tensor-field shape-operation surface.

Every method round-trips through a snap holding one nested dataclass field
(``device``), which is how nested recursion is pinned in each case; a
dedicated absent-optional-field snap pins the skip-``None`` rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common import TensorGroupMixin, walk_tensor_fields


@dataclass(frozen=True)
class _DeviceSnap(TensorGroupMixin):
    """Nested per-device snap."""

    g__uS: Tensor


@dataclass(frozen=True)
class _CellSnap(TensorGroupMixin):
    """Cell-grid snap with one nested dataclass field and one plain field."""

    v_wl__V: Tensor
    device: _DeviceSnap
    label: str


@dataclass(frozen=True)
class _LinearCellSnap(_CellSnap):
    """A leaf subtype, to pin type-preservation through the mixin."""

    g_chord__uS: Tensor


@dataclass(frozen=True)
class _OptionalSnap(TensorGroupMixin):
    """Snap with one absent-capable field."""

    v_wl__V: Tensor
    bias__V: Tensor | None


# --- map_tensors ---


def test_map_tensors_applies_fn_to_every_field_and_recurses() -> None:
    snap = _CellSnap(v_wl__V=torch.tensor([1.0, 2.0]), device=_DeviceSnap(g__uS=torch.tensor([3.0])), label="row")

    doubled = snap.map_tensors(lambda t: t * 2.0)

    torch.testing.assert_close(doubled.v_wl__V, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(doubled.device.g__uS, torch.tensor([6.0]))
    assert doubled.label == "row"
    # The input is left as it was.
    torch.testing.assert_close(snap.v_wl__V, torch.tensor([1.0, 2.0]))


def test_map_tensors_skips_an_absent_optional_field() -> None:
    snap = _OptionalSnap(v_wl__V=torch.zeros(2), bias__V=None)

    mapped = snap.map_tensors(lambda t: t + 1.0)

    assert mapped.bias__V is None
    torch.testing.assert_close(mapped.v_wl__V, torch.ones(2))


def test_map_tensors_preserves_the_concrete_subclass() -> None:
    snap = _LinearCellSnap(
        v_wl__V=torch.zeros(2),
        device=_DeviceSnap(g__uS=torch.zeros(2)),
        label="row",
        g_chord__uS=torch.ones(2),
    )

    mapped = snap.map_tensors(lambda t: t + 1.0)

    assert type(mapped) is _LinearCellSnap
    torch.testing.assert_close(mapped.g_chord__uS, torch.full((2,), 2.0))


def test_map_tensors_is_walk_tensor_fields_applied_to_self() -> None:
    snap = _CellSnap(v_wl__V=torch.ones(2), device=_DeviceSnap(g__uS=torch.ones(2)), label="row")

    via_mixin = snap.map_tensors(lambda t: t * 3.0)
    via_primitive = walk_tensor_fields(snap, lambda t: t * 3.0)

    torch.testing.assert_close(via_mixin.v_wl__V, via_primitive.v_wl__V)
    torch.testing.assert_close(via_mixin.device.g__uS, via_primitive.device.g__uS)
    assert via_mixin.label == via_primitive.label


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

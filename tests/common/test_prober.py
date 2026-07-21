"""Tests for the probe side channel: per-subclass isolation, stacking, gating.

A probe record is a detached payload alone — no emitting module or name is
stored. Each concrete :class:`Prober` subclass binds one observation link and
carries its own active stack, so a payload submitted to one subclass never
reaches an active prober of another. That isolation is the link routing that
replaces channel filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

import pytest
import torch
from torch import Tensor

from neurox.common.prober import Prober


@dataclass(frozen=True)
class _Payload:
    """A minimal probe payload that records whether ``detach`` ran."""

    code: Tensor
    detached: bool = False

    def detach(self) -> _Payload:
        return replace(self, code=self.code.detach(), detached=True)


class _ProberA(Prober[_Payload]):
    """First link's capture point."""

    _active_stack: ClassVar[list[Prober[_Payload]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[_Payload]]:
        return cls._active_stack


class _ProberB(Prober[_Payload]):
    """Second, independent link's capture point."""

    _active_stack: ClassVar[list[Prober[_Payload]]] = []

    @classmethod
    def _stack(cls) -> list[Prober[_Payload]]:
        return cls._active_stack


def test_subclass_stacks_are_isolated() -> None:
    """A payload submitted to subclass A never reaches an active B prober."""
    with _ProberA() as a, _ProberB() as b:
        _ProberA.submit(_Payload(code=torch.tensor([1.0])))
        _ProberB.submit(_Payload(code=torch.tensor([2.0])))
    assert [r.code.item() for r in a.records] == [1.0]
    assert [r.code.item() for r in b.records] == [2.0]


def test_submit_stores_detached_payloads_in_order() -> None:
    """Payloads keep submission order and are stored via ``detach``."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    with _ProberA() as prober:
        _ProberA.submit(_Payload(code=x))
        _ProberA.submit(_Payload(code=torch.tensor([3.0])))
    records = prober.records
    assert [r.detached for r in records] == [True, True]
    assert not records[0].code.requires_grad
    assert records[0].code.grad_fn is None
    assert torch.equal(records[0].code, torch.tensor([1.0, 2.0]))
    assert torch.equal(records[1].code, torch.tensor([3.0]))


def test_no_active_prober_is_a_no_op() -> None:
    """Submitting without a session returns before touching the payload."""
    _ProberA.submit(_Payload(code=torch.tensor([1.0])))  # must not raise
    assert _ProberA.active() is False


def test_stack_nesting_routes_to_every_active_prober() -> None:
    """An emission reaches every stacked prober; a popped prober stops receiving."""
    with _ProberA() as outer:
        _ProberA.submit(_Payload(code=torch.tensor([1.0])))
        with _ProberA() as inner:
            _ProberA.submit(_Payload(code=torch.tensor([2.0])))
        _ProberA.submit(_Payload(code=torch.tensor([3.0])))
    assert [r.code.item() for r in outer.records] == [1.0, 2.0, 3.0]
    assert [r.code.item() for r in inner.records] == [2.0]
    assert _ProberA._active_stack == []


def test_nested_same_subclass_probers_share_one_detached_object() -> None:
    """Stacked probers of one subclass hold the SAME payload object, detached once."""
    with _ProberA() as outer, _ProberA() as inner:
        _ProberA.submit(_Payload(code=torch.tensor([1.0])))
    assert outer.records[0] is inner.records[0]
    assert outer.records[0].detached is True


def test_active_lifecycle() -> None:
    """``active`` is False before entry, True inside, False after exit."""
    assert _ProberA.active() is False
    with _ProberA():
        assert _ProberA.active() is True
    assert _ProberA.active() is False


def test_active_is_per_subclass() -> None:
    """An active A prober does not make B active."""
    with _ProberA():
        assert _ProberA.active() is True
        assert _ProberB.active() is False


def test_exit_restores_the_stack_on_exception() -> None:
    """A raising body still pops the prober's own frame."""
    with pytest.raises(RuntimeError, match="boom"), _ProberA():
        raise RuntimeError("boom")
    assert _ProberA._active_stack == []


def test_active_on_abstract_base_raises() -> None:
    """The abstract base provides no concrete active stack."""
    with pytest.raises(NotImplementedError):
        Prober.active()


def test_submit_on_abstract_base_raises() -> None:
    """The abstract base cannot route a submitted payload."""
    with pytest.raises(NotImplementedError):
        Prober.submit(_Payload(code=torch.tensor([1.0])))


def test_abstract_base_cannot_be_instantiated() -> None:
    """The abstract :class:`Prober` base is not instantiable."""
    with pytest.raises(TypeError):
        Prober()

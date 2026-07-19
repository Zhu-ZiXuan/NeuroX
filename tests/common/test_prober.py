"""Tests for the probe side channel: stack nesting, allowlist gating, alignment.

A probe record carries its emitting module and detached tensors; naming is
resolved only against a root, exactly as the profiler does it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import ProbeMixin
from neurox.common.prober import AdcProber, Prober


class _Emitter(nn.Module, ProbeMixin):
    """Minimal emitting host: same base pairing as ``ModuleBase``."""

    def emit(self, channel: str, **tensors: Tensor) -> None:
        self._probe_record(channel, **tensors)


class _Owner(nn.Module):
    def __init__(self, emitter: _Emitter) -> None:
        super().__init__()
        self.emitter = emitter


def test_submit_stores_detached_records_in_order() -> None:
    """Records keep submission order; stored tensors are detached views."""
    emitter = _Emitter()
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    with Prober() as prober:
        emitter.emit("ch", code=x)
        emitter.emit("ch", code=torch.tensor([3.0]))
    records = prober.records("ch")
    assert [module for module, _ in records] == [emitter, emitter]
    assert not records[0][1]["code"].requires_grad
    assert torch.equal(records[0][1]["code"], torch.tensor([1.0, 2.0]))
    assert torch.equal(records[1][1]["code"], torch.tensor([3.0]))


def test_no_active_prober_is_a_no_op() -> None:
    """Without a session the hook returns before touching any tensor."""
    emitter = _Emitter()
    emitter.emit("ch", code=torch.tensor([1.0]))  # must not raise
    with Prober() as prober:
        pass
    assert prober.records("ch") == []


def test_stack_nesting_routes_to_every_active_prober() -> None:
    """An emission reaches every stacked prober; a popped prober stops receiving."""
    emitter = _Emitter()
    with Prober() as outer:
        emitter.emit("ch", code=torch.tensor([1.0]))
        with Prober() as inner:
            emitter.emit("ch", code=torch.tensor([2.0]))
        emitter.emit("ch", code=torch.tensor([3.0]))
    assert [t["code"].item() for _, t in outer.records("ch")] == [1.0, 2.0, 3.0]
    assert [t["code"].item() for _, t in inner.records("ch")] == [2.0]
    assert Prober._active_stack == []


def test_channel_allowlist_gates_submission() -> None:
    """A prober stores only allowlisted channels; ``None`` accepts everything."""
    emitter = _Emitter()
    with Prober(channels=frozenset({"keep"})) as gated, Prober() as open_prober:
        emitter.emit("keep", code=torch.tensor([1.0]))
        emitter.emit("drop", code=torch.tensor([2.0]))
    assert len(gated.records("keep")) == 1
    assert gated.records("drop") == []
    assert len(open_prober.records("keep")) == 1
    assert len(open_prober.records("drop")) == 1


def test_stacked_stacks_one_key_across_records() -> None:
    """``stacked`` adds a leading record axis in submission order."""
    emitter = _Emitter()
    with Prober() as prober:
        emitter.emit("ch", code=torch.tensor([1, 2]))
        emitter.emit("ch", code=torch.tensor([3, 4]))
    assert torch.equal(prober.stacked("ch", "code"), torch.tensor([[1, 2], [3, 4]]))
    with pytest.raises(ValueError, match="no records"):
        prober.stacked("empty", "code")


def test_paired_aligns_records_by_order() -> None:
    """``paired`` zips two channels positionally and rejects a length mismatch."""
    a, b = _Emitter(), _Emitter()
    with Prober() as prober:
        a.emit("ch.a", code=torch.tensor([1.0]))
        b.emit("ch.b", code=torch.tensor([10.0]))
        a.emit("ch.a", code=torch.tensor([2.0]))
        b.emit("ch.b", code=torch.tensor([20.0]))
    pairs = prober.paired("ch.a", "ch.b")
    assert [(ra[1]["code"].item(), rb[1]["code"].item()) for ra, rb in pairs] == [(1.0, 10.0), (2.0, 20.0)]
    with Prober() as lopsided:
        a.emit("ch.a", code=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="record counts differ"):
        lopsided.paired("ch.a", "ch.b")


def test_resolve_names_maps_id_to_qualified_name() -> None:
    """The name map comes from the root's traversal; the root itself is ``""``."""
    emitter = _Emitter()
    owner = _Owner(emitter)
    names = Prober.resolve_names(owner)
    assert names[id(emitter)] == "emitter"
    assert names[id(owner)] == ""


def test_adc_prober_defaults_to_the_adc_channels() -> None:
    """``AdcProber()`` admits exactly the two ADC channels; views read them."""
    emitter = _Emitter()
    with AdcProber() as prober:
        emitter.emit(AdcProber.ADC_CONVERT, code=torch.tensor([1]))
        emitter.emit(AdcProber.ADC_IDEAL_VMM, code=torch.tensor([2]))
        emitter.emit("other", code=torch.tensor([3]))
    assert prober.channels == frozenset({"adc.convert", "adc.ideal_vmm"})
    assert len(prober.convert_records()) == 1
    assert len(prober.ideal_vmm_records()) == 1
    assert prober.records("other") == []
    ((_, convert), (_, ideal)) = prober.paired_conversions()[0]
    assert convert["code"].item() == 1
    assert ideal["code"].item() == 2


def test_exit_restores_the_stack_on_exception() -> None:
    """A raising body still pops the prober's own frame."""
    with pytest.raises(RuntimeError, match="boom"), Prober():
        raise RuntimeError("boom")
    assert Prober._active_stack == []

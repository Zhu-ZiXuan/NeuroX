"""R25 fabricate-contract regression for the ``XbarArray`` ABC.

The ``XbarArray`` ABC sits between :class:`FabricateMixin` and the concrete
:class:`XbarArray1t1r`. It owns no static state of its own, so it must supply
``_sample_fabricate_mismatch`` as an explicit no-op; if it forgot it, either the
abstract method would re-raise or a spurious body would perturb the once-per-node
resample. These tests pin that ``fabricate()`` on an ``XbarArray1t1r`` resamples
every fabricable node's static state EXACTLY ONCE in pre-order, and that the ABC
override is a genuine no-op the concrete array inherits unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from neurox.common.mixin import FabricateMixin
from neurox.primitive.device import MosfetPolicy, RramPolicy
from neurox.primitive.device.mosfet import Nmos
from neurox.primitive.device.rram import Rram
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rPolicy
from neurox.primitive.xbar.array.base import XbarArray
from neurox.primitive.xbar.cell import XbarCell1t1r, XbarCell1t1rPolicy
from works.offset_1t1r.macro import Offset1t1rCimMacroConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
CHIP_CONFIG = REPO_ROOT / "works" / "offset_1t1r" / "config" / "1t1r_28nm.toml"


def _build_array(*, mismatch: bool, device: torch.device) -> XbarArray1t1r:
    """Build a small standalone 1T1R pure array from the chip preset.

    Reads only ``[cim_macro]`` for the owned ``core_config`` (an
    ``XbarArray1t1rConfig``); the composite policy is constructed with the
    device-mismatch toggles set from ``mismatch`` so the fabricate cascade has
    real static state to resample.
    """
    macro_config = Offset1t1rCimMacroConfig.from_file(CHIP_CONFIG, section="cim_macro")
    core_config = macro_config.array_config
    policy = XbarArray1t1rPolicy(
        cell=XbarCell1t1rPolicy(
            rram=RramPolicy(prog_gamma=mismatch, stuck_at=mismatch, read_telegraph=False, read_thermal=False),
            nmos=MosfetPolicy(A_vt_mismatch=mismatch, A_beta_mismatch=mismatch),
        ),
        solve_chunk_size=0,
    )
    array = XbarArray1t1r(
        config=core_config,
        policy=policy,
        w_layout_shape=(8, 8),
        dtype=torch.float64,
        T__K=300.0,
    )
    array.to(device)
    array.eval()
    return array


def _fabricable_tree(node: FabricateMixin) -> Iterator[FabricateMixin]:
    """Yield ``node`` then every fabricable descendant in pre-order."""
    yield node
    for child in node._fabricable_children():
        yield from _fabricable_tree(child)


def test_xbar_array_abc_supplies_noop_sample_fabricate_mismatch(device: torch.device) -> None:
    """The ABC owns the no-op; the concrete 1T1R array does not override it."""
    assert "_sample_fabricate_mismatch" not in XbarArray1t1r.__dict__
    assert "_sample_fabricate_mismatch" in XbarArray.__dict__
    assert XbarArray1t1r._sample_fabricate_mismatch is XbarArray._sample_fabricate_mismatch

    # And it is a genuine no-op: returns None and touches no state.
    array = _build_array(mismatch=False, device=device)
    before = {name: buf.clone() for name, buf in array.named_buffers()}
    array._sample_fabricate_mismatch()  # no-op: must neither raise nor mutate state
    after = dict(array.named_buffers())
    assert before.keys() == after.keys()
    for name, buf in before.items():
        assert torch.equal(buf, after[name])


def test_array_fabricate_resamples_each_node_once_preorder(device: torch.device) -> None:
    """``fabricate()`` visits every fabricable node exactly once, pre-order."""
    array = _build_array(mismatch=True, device=device)

    # Snapshot the true tree BEFORE patching so traversal is untouched.
    nodes = list(_fabricable_tree(array))

    # The cell/array split must still expose the cell + its RRAM / NMOS as
    # fabricable descendants of the array.
    node_types = {type(n) for n in nodes}
    assert XbarArray1t1r in node_types
    assert XbarCell1t1r in node_types
    assert Rram in node_types
    assert Nmos in node_types

    order: list[FabricateMixin] = []
    counts: dict[int, int] = {id(n): 0 for n in nodes}

    for node in nodes:
        original: Callable[[], None] = node._sample_fabricate_mismatch

        def make_spy(n: FabricateMixin, orig: Callable[[], None]) -> Callable[[], None]:
            def spy() -> None:
                order.append(n)
                counts[id(n)] += 1
                orig()

            return spy

        # Instance-level shadow of the bound method; class methods untouched.
        node._sample_fabricate_mismatch = make_spy(node, original)  # type: ignore[method-assign]

    array.fabricate()

    # Exactly once per node, and no node missed.
    assert order and len(order) == len(nodes)
    assert all(count == 1 for count in counts.values())

    # Pre-order: every parent is sampled strictly before each of its children.
    position = {id(n): i for i, n in enumerate(order)}
    for parent in nodes:
        for child in parent._fabricable_children():
            assert position[id(parent)] < position[id(child)]

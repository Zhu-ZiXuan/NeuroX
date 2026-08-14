"""Fabrication traversal tests for a concrete 1T1R array tree.

The traversal test verifies that `fabricate()` reaches every fabricable node
exactly once in the documented pre-order. All policies are off; the spy counts
sampling calls regardless of whether a perturbation is enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch

from neurox.common.fabricate_mixin import FabricateMixin
from neurox.primitive.device import MosfetConfig, MosfetPolicy, Nmos, Rram, RramConfig, RramPolicy
from neurox.primitive.xbar.array import (
    XbarArray1t1r,
    XbarArray1t1rConfig,
    XbarArray1t1rOperationMode,
    XbarArray1t1rPolicy,
)
from neurox.primitive.xbar.cell import XbarCell1t1rDetail, XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy
from neurox.primitive.xbar.solver import NestedParallelRailSolverConfig


def _array_config() -> XbarArray1t1rConfig:
    """Hand-written tiny array config; only the device configs come from
    the library presets (the sanctioned device exception)."""
    rram_config = RramConfig.from_preset("process/rram:default")
    cell_config = XbarCell1t1rDetailConfig(
        rram_config=rram_config,
        nmos_config=MosfetConfig.from_preset("process/mos:nmos_28_rvt"),
        state_to_g_map__uS=(rram_config.g_min__uS, 100.0),
        access_nmos_W__um=0.1,
        access_nmos_L__um=0.05,
        rram_g_max__uS=100.0,
        newton_iter_num=2,
    )
    return XbarArray1t1rConfig(
        row_cell_space__um=1.0,
        col_cell_space__um=1.0,
        bl_segment_r__MOhm=1e-4,
        sl_segment_r__MOhm=1e-4,
        bl_node_c__fF=0.1,
        x_node_c__fF=0.1,
        sl_node_c__fF=0.1,
        wl_node_c__fF=0.1,
        cell_config=cell_config,
        solver_config=NestedParallelRailSolverConfig(n_outer=1, n_inner=1),
    )


def _build_array(*, device: torch.device) -> XbarArray1t1r:
    """Build a minimal standalone 1T1R pure array, every policy toggle off."""
    policy = _array_policy(solve_chunk_size=0)
    array = XbarArray1t1r(
        config=_array_config(),
        policy=policy,
        inst_shape=(),
        row_num=2,
        col_num=2,
        operation_mode=XbarArray1t1rOperationMode.WL_IN_BL_SCAN,
        v_dd_wl__V=1.0,
        v_dd_bl__V=1.0,
        dtype=torch.float64,
        T__K=300.0,
    )
    array.to(device)
    array.eval()
    return array


def _array_policy(*, solve_chunk_size: int) -> XbarArray1t1rPolicy:
    """Build an all-off policy with the requested solver chunk size."""
    return XbarArray1t1rPolicy(
        cell_policy=XbarCell1t1rDetailPolicy(
            rram_policy=RramPolicy(
                prog_gamma=False,
                drift=False,
                stuck_at=False,
                read_telegraph=False,
                read_thermal=False,
            ),
            nmos_policy=MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False),
        ),
        solve_chunk_size=solve_chunk_size,
    )


def _fabricable_tree(node: FabricateMixin) -> Iterator[FabricateMixin]:
    """Yield `node` then every fabricable descendant in pre-order."""
    yield node
    for child in node._fabricable_children():
        yield from _fabricable_tree(child)


def test_xbar_array_policy_rejects_negative_chunk_size() -> None:
    with pytest.raises(ValueError, match="solve_chunk_size"):
        _array_policy(solve_chunk_size=-1)


def test_array_fabricate_resamples_each_node_once_preorder(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fabricate()` visits every fabricable node exactly once, pre-order."""
    array = _build_array(device=device)

    # Snapshot the true tree BEFORE patching so traversal is untouched.
    nodes = list(_fabricable_tree(array))

    # The cell/array split must still expose the cell + its RRAM / NMOS as
    # fabricable descendants of the array.
    node_types = {type(n) for n in nodes}
    assert XbarArray1t1r in node_types
    assert XbarCell1t1rDetail in node_types
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
        monkeypatch.setattr(node, "_sample_fabricate_mismatch", make_spy(node, original))

    array.fabricate()

    # Exactly once per node, and no node missed.
    assert order
    assert len(order) == len(nodes)
    assert all(count == 1 for count in counts.values())

    # Pre-order: every parent is sampled strictly before each of its children.
    position = {id(n): i for i, n in enumerate(order)}
    for parent in nodes:
        for child in parent._fabricable_children():
            assert position[id(parent)] < position[id(child)]

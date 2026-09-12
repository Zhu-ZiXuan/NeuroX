"""Array fabrication visits each registered module once in pre-order."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from neurox.common.module import ModuleBase
from neurox.primitive.analog import VoltageDriver, VoltageDriverConfig, VoltageDriverPolicy, VoltageDriverSnap
from neurox.primitive.device.mosfet import MosfetConfig, MosfetPolicy, Nmos
from neurox.primitive.device.rram import Rram, RramConfig, RramPolicy
from neurox.primitive.xbar.array import XbarArray1t1r, XbarArray1t1rConfig, XbarArray1t1rPolicy
from neurox.primitive.xbar.cell import XbarCell1t1rDetail, XbarCell1t1rDetailConfig, XbarCell1t1rDetailPolicy

type _Array = XbarArray1t1r[VoltageDriverSnap, VoltageDriverSnap]
type _Module = ModuleBase


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
    )


def _build_array(*, device: torch.device) -> _Array:
    """Build a minimal standalone 1T1R pure array, every policy toggle off."""
    policy = _array_policy(solve_chunk_size=0)
    array = XbarArray1t1r(
        bl_driver=_driver(device=device),
        sl_driver=_driver(device=device),
        config=_array_config(),
        policy=policy,
        inst_shape=(),
        row_num=2,
        col_num=2,
        vdd__V=1.0,
        dtype=torch.float64,
        T__K=300.0,
    )
    array.to(device)
    array.eval()
    return array


def _driver(*, device: torch.device) -> VoltageDriver:
    return VoltageDriver(
        config=VoltageDriverConfig(
            r_out__MOhm=0.0,
            offset_sigma__V=0.0,
            thermal_sigma__V=0.0,
            energy_per_op__fJ=0.0,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
        ),
        policy=VoltageDriverPolicy(offset=False, thermal=False),
        inst_shape=(2,),
        dtype=torch.float64,
        T__K=300.0,
    ).to(device)


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


def test_xbar_array_policy_rejects_negative_chunk_size() -> None:
    with pytest.raises(ValueError, match="solve_chunk_size"):
        _array_policy(solve_chunk_size=-1)


def test_array_fabricate_resamples_each_node_once_preorder(
    device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fabricate()` visits every fabricable node exactly once, pre-order."""
    array = _build_array(device=device)

    # Snapshot the true tree BEFORE patching so traversal is untouched.
    nodes = [node for node in array.modules() if isinstance(node, ModuleBase)]

    node_types = {type(n) for n in nodes}
    assert XbarArray1t1r in node_types
    assert XbarCell1t1rDetail in node_types
    assert Rram in node_types
    assert Nmos in node_types

    order: list[_Module] = []
    counts: dict[int, int] = {id(n): 0 for n in nodes}

    for node in nodes:
        original: Callable[[], None] = node._sample_fabrication_variation

        def make_spy(n: _Module, orig: Callable[[], None]) -> Callable[[], None]:
            def spy() -> None:
                order.append(n)
                counts[id(n)] += 1
                orig()

            return spy

        # Instance-level shadow of the bound method; class methods untouched.
        monkeypatch.setattr(node, "_sample_fabrication_variation", make_spy(node, original))

    array.fabricate()

    # Exactly once per node, and no node missed.
    assert order
    assert order == nodes
    assert all(count == 1 for count in counts.values())

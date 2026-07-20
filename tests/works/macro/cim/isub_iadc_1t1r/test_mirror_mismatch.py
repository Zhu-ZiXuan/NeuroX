"""Fabricate-once mismatch regression for the p / n kernel mirror stages.

The two mirror stages are column-MUX time-shared kernel
:class:`~neurox.primitive.analog.CurrentMirror` blocks whose fabricated
buffers live at the REAL device count with a trailing size-1 broadcast axis
(the reshape-broadcast sharing trick): front-end ``(2, n_lane, 1)`` against a
``[..., 2, n_lane, col/n_lane]`` forward tensor, back-end ``(2, n_io, 1)``.
P and N polarities conduct through separate devices and must carry
INDEPENDENT static mismatch draws. This suite builds the full tile at the
tiny overlay geometry (``n_lane = 2``, ``n_io = 1``) and asserts:

  (i)   under ``all_off`` the chain is bit-exact: repeated VMMs, repeated
        analog probes, and an independently refabricated tile all agree, and
        the mirror buffers are exactly the unit ratio;
  (ii)  a NONZERO sigma with the policy OFF still leaves the exact unit ratio
        (the policy gates, the config only sizes);
  (iii) with ``mismatch`` ON and sigma > 0 the fabricated buffers depart from
        unity at the real device shape, the analog chain output changes, two
        eval-mode calls on one fabricate stay bit-identical (no per-call
        draw), and two fabricates draw differently;
  (iv)  the P and N polarity slices of one stage differ (independent draws),
        and the p-stage / n-stage draws are mutually independent.

Eager (dynamo disabled), device-threaded; a few seconds of numerics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.analog import CurrentMirrorPolicy
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    CONFIG_PATH,
    TINY_COL_NUM,
    TINY_OVERLAY_PATH,
    TINY_ROW_NUM,
    array_read,
    build_tile,
    load_all_off_policy,
    load_config,
    probe_i_sub,
    tile_device,
)

_SIGMA = 0.2


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build(
    device: torch.device,
    *,
    p_sigma: float = 0.0,
    n_sigma: float = 0.0,
    p_mismatch: bool = False,
    n_mismatch: bool = False,
    seed: int | None = None,
) -> IsubIadc1t1rCimMacro:
    """Build + fabricate the tiny tile with mirror sigma / policy overrides."""
    config = load_config(TINY_OVERLAY_PATH, CONFIG_PATH)
    config = dataclasses.replace(
        config,
        p_mirror_config=dataclasses.replace(config.p_mirror_config, ratio_sigma_relative=p_sigma),
        n_mirror_config=dataclasses.replace(config.n_mirror_config, ratio_sigma_relative=n_sigma),
    )
    policy = dataclasses.replace(
        load_all_off_policy(),
        p_mirror=CurrentMirrorPolicy(mismatch=p_mismatch),
        n_mirror=CurrentMirrorPolicy(mismatch=n_mismatch),
    )
    return build_tile(config, device=device, policy=policy, seed=seed)


def _probe_chain(xbar: IsubIadc1t1rCimMacro) -> tuple[Tensor, Tensor]:
    """Deterministic analog probe: (i_wdl, i_sub) for a fixed +1-column read.

    Programs logical column 0 all ``+1`` and drives every word line of one
    single WL plane, then reproduces the analog chain up to the ADC input
    (:func:`probe_i_sub` also returns the p-mirror output via a second call
    point). NOTE: reprograms the tile.
    """
    dev = tile_device(xbar)
    w = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long, device=dev)
    w[0, 0, :] = 1
    xbar.program(w)
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long, device=dev)

    with torch.no_grad():
        steady = array_read(xbar, x)
        i_pol = steady.i_bl_port__uA.unflatten(-1, (xbar.col_num, 2)).movedim(-1, -2)
        i_lane = xbar._split_col_lanes(i_pol, col_per_lane=xbar.config.mux_factor)
        i_wdl = xbar.p_mirror.replicate(i_lane)
    i_sub, _sign = probe_i_sub(xbar, x)
    return i_wdl, i_sub


def test_all_off_bit_exact(device: torch.device) -> None:
    """all_off: unit buffers; repeated probes and a refabricated tile agree bit-exactly."""
    xbar = _build(device)

    # Fabricated mirror buffers are exactly the unit ratio.
    assert torch.equal(xbar.p_mirror.ratio_mismatch, torch.ones_like(xbar.p_mirror.ratio_mismatch))
    assert torch.equal(xbar.n_mirror.ratio_mismatch, torch.ones_like(xbar.n_mirror.ratio_mismatch))

    i_wdl_1, i_sub_1 = _probe_chain(xbar)
    i_wdl_2, i_sub_2 = _probe_chain(xbar)
    assert torch.equal(i_wdl_1, i_wdl_2)
    assert torch.equal(i_sub_1, i_sub_2)

    # An independent rebuild + refabricate reproduces the same analog chain.
    rebuilt = _build(device)
    i_wdl_3, i_sub_3 = _probe_chain(rebuilt)
    assert torch.equal(i_wdl_1, i_wdl_3)
    assert torch.equal(i_sub_1, i_sub_3)


def test_nonzero_sigma_policy_off_is_exact(device: torch.device) -> None:
    """A nonzero sigma with the policy OFF leaves the exact ratio copy (gating)."""
    gated = _build(device, p_sigma=_SIGMA, n_sigma=_SIGMA, p_mismatch=False, n_mismatch=False, seed=20200709)
    nominal = _build(device)

    assert torch.equal(gated.p_mirror.ratio_mismatch, torch.ones_like(gated.p_mirror.ratio_mismatch))
    assert torch.equal(gated.n_mirror.ratio_mismatch, torch.ones_like(gated.n_mirror.ratio_mismatch))

    i_wdl_g, i_sub_g = _probe_chain(gated)
    i_wdl_n, i_sub_n = _probe_chain(nominal)
    assert torch.equal(i_wdl_g, i_wdl_n)
    assert torch.equal(i_sub_g, i_sub_n)


def test_mismatch_perturbs_chain_and_repeats(device: torch.device) -> None:
    """mismatch ON: buffers depart from unity, the chain changes, eval calls repeat."""
    torch.manual_seed(20200709)
    baseline = _build(device)
    i_wdl_base, i_sub_base = _probe_chain(baseline)

    on = _build(device, p_sigma=_SIGMA, n_sigma=_SIGMA, p_mismatch=True, n_mismatch=True, seed=20200709)

    # The fabricated buffers live at the REAL shared device shape (trailing
    # size-1 broadcast axis) and depart from unity.
    assert tuple(on.p_mirror.ratio_mismatch.shape) == (2, on.n_lane, 1)
    assert tuple(on.n_mirror.ratio_mismatch.shape) == (2, on.n_io, 1)
    assert not torch.equal(on.p_mirror.ratio_mismatch, torch.ones_like(on.p_mirror.ratio_mismatch))
    assert not torch.equal(on.n_mirror.ratio_mismatch, torch.ones_like(on.n_mirror.ratio_mismatch))

    # The analog chain output changes vs the all_off baseline.
    i_wdl_on, i_sub_on = _probe_chain(on)
    assert not torch.equal(i_wdl_on, i_wdl_base)
    assert not torch.equal(i_sub_on, i_sub_base)

    # Two eval-mode probes on ONE fabricate are bit-identical (no per-call draw).
    i_wdl_again, i_sub_again = _probe_chain(on)
    assert torch.equal(i_wdl_on, i_wdl_again)
    assert torch.equal(i_sub_on, i_sub_again)

    # A second fabricate draws a DIFFERENT static mismatch.
    p_first = on.p_mirror.ratio_mismatch.clone()
    n_first = on.n_mirror.ratio_mismatch.clone()
    on.fabricate()
    assert not torch.equal(on.p_mirror.ratio_mismatch, p_first)
    assert not torch.equal(on.n_mirror.ratio_mismatch, n_first)


def test_p_and_n_mismatch_independent(device: torch.device) -> None:
    """P vs N polarity slices differ, and the p-stage / n-stage draws differ."""
    on = _build(device, p_sigma=_SIGMA, n_sigma=_SIGMA, p_mismatch=True, n_mismatch=True, seed=20200709)

    # Within each stage: the leading polarity axis carries independent draws.
    p_buf = on.p_mirror.ratio_mismatch  # (2, n_lane = 2, 1)
    n_buf = on.n_mirror.ratio_mismatch  # (2, n_io = 1, 1)
    assert not torch.equal(p_buf[0], p_buf[1]), "P and N front-end devices must draw independently"
    assert not torch.equal(n_buf[0], n_buf[1]), "P and N back-end devices must draw independently"

    # Across stages: the front-end and back-end draws are independent
    # (compare the first lane of each polarity against the back-end device).
    assert not torch.equal(p_buf[:, :1, :], n_buf)

    # Enabling only the p-stage leaves the n-stage at the exact unit ratio.
    p_only = _build(device, p_sigma=_SIGMA, n_sigma=_SIGMA, p_mismatch=True, n_mismatch=False, seed=20200709)
    assert not torch.equal(p_only.p_mirror.ratio_mismatch, torch.ones_like(p_only.p_mirror.ratio_mismatch))
    assert torch.equal(p_only.n_mirror.ratio_mismatch, torch.ones_like(p_only.n_mirror.ratio_mismatch))

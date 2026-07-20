"""Fabricate-once mismatch regression for the kernel subtractor back-end.

The subtractor is a per-CIM-IO kernel
:class:`~neurox.primitive.analog.CurrentSubtractor` whose static buffers live
at ``(n_io, 1)`` (the real shared device count with the trailing broadcast
axis). Two static nonidealities gate on it: the subtracted-leg ratio
``mismatch`` (multiplicative, perturbs the recovered magnitude) and the
sign-comparator ``offset`` (additive input-referred current, can flip the
decoded sign where the polarity difference is small). This suite builds the
full tile at the tiny overlay geometry and asserts on the PHASE-0 codes
(the tested weight rows live in active phase 0):

  (i)   NONZERO sigmas with both policies OFF leave the exact identity
        (unit ratio, zero offset) and a bit-exact all_off chain;
  (ii)  ``offset`` ON with a sigma comparable to the per-MAC-unit current
        flips the decoded sign of a ``|M| = 1`` column for SOME fabrication
        draws and leaves it correct for others (seed-swept, both outcomes
        must occur), and perturbs the ``M = 0`` balance point to nonzero
        codes of either sign;
  (iii) ``mismatch`` ON perturbs the analog ``I_SUB`` magnitude
        deterministically, decodes differently from ideal for some draws on
        a saturating ``-1`` column (the subtracted leg carries the ratio),
        and two eval-mode calls on one fabricate stay bit-identical.

Eager (dynamo disabled), device-threaded; a few seconds of numerics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
import torch._dynamo
from torch import Tensor

from neurox.primitive.analog import CurrentSubtractorPolicy
from neurox.works.macro.cim.isub_iadc_1t1r.macro import IsubIadc1t1rCimMacro
from tests.works.macro.cim.isub_iadc_1t1r._utils import (
    ADC_OP,
    CONFIG_PATH,
    MAG_MAX,
    TINY_COL_NUM,
    TINY_OVERLAY_PATH,
    TINY_ROW_NUM,
    build_tile,
    load_all_off_policy,
    load_config,
    masked_planes,
    probe_i_sub,
    tile_device,
)

# Offset sigma [uA] comparable to the ~1.65 uA per-MAC-unit I_SUB step at the
# ADC input (the tiny-tile I_SUB(M) grid), so a |M| = 1 sign flip is a likely
# (not certain) draw.
_OFFSET_SIGMA__uA = 5.0
# Relative ratio sigma on the subtracted leg.
_MISMATCH_SIGMA = 0.3
# Seed sweep for the statistical (seed-fixed) assertions.
_SEEDS = range(24)


@pytest.fixture(autouse=True)
def _eager() -> Iterator[None]:
    """Run eagerly — the solver leaf is ``@torch.compile``; do not unroll it."""
    with torch._dynamo.config.patch(disable=True):
        yield


def _build(
    device: torch.device,
    *,
    mismatch_sigma: float = 0.0,
    offset_sigma__uA: float = 0.0,
    mismatch: bool = False,
    offset: bool = False,
    seed: int | None = None,
) -> IsubIadc1t1rCimMacro:
    """Build + fabricate the tiny tile with subtractor sigma / policy overrides."""
    config = load_config(TINY_OVERLAY_PATH, CONFIG_PATH)
    config = dataclasses.replace(
        config,
        subtractor_config=dataclasses.replace(
            config.subtractor_config,
            mismatch_sigma_relative=mismatch_sigma,
            offset_sigma__uA=offset_sigma__uA,
        ),
    )
    policy = dataclasses.replace(
        load_all_off_policy(), subtractor=CurrentSubtractorPolicy(mismatch=mismatch, offset=offset)
    )
    return build_tile(config, device=device, policy=policy, seed=seed)


def _decode_col0_phase0(xbar: IsubIadc1t1rCimMacro, w_col0: Tensor, x: Tensor) -> int:
    """Program ``w_col0`` into logical column 0 (others 0), decode phase 0 of column 0."""
    dev = tile_device(xbar)
    w = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long, device=dev)
    w[0, 0, :] = w_col0.to(dev)
    xbar.program(w)
    # Shape: [row_num] -> [P, row_num]  (caller-side sub-phase expansion)
    planes = masked_planes(x.to(dev), row_num=TINY_ROW_NUM, max_active_rows=xbar.max_active_rows)
    with torch.no_grad():
        out = xbar.vec_mat_mul(planes, adc_operation_point=ADC_OP)
    return int(out[0, 0])  # [P, col] -> phase 0, column 0


def _probe_i_sub_col0(xbar: IsubIadc1t1rCimMacro, w_col0: Tensor, x: Tensor) -> Tensor:
    """Analog ``I_SUB`` at the ADC input for column-0 pattern ``w_col0`` under ``x``."""
    dev = tile_device(xbar)
    w = torch.zeros((TINY_COL_NUM, 1, TINY_ROW_NUM), dtype=torch.long, device=dev)
    w[0, 0, :] = w_col0.to(dev)
    xbar.program(w)
    # Shape: [row_num] -> [P, row_num]  (caller-side sub-phase expansion)
    planes = masked_planes(x.to(dev), row_num=TINY_ROW_NUM, max_active_rows=xbar.max_active_rows)
    i_sub, _sign = probe_i_sub(xbar, planes)
    return i_sub


def test_nonzero_sigma_policy_off_is_identity(device: torch.device) -> None:
    """Nonzero sigmas with both policies OFF leave unit ratio / zero offset."""
    gated = _build(
        device,
        mismatch_sigma=_MISMATCH_SIGMA,
        offset_sigma__uA=_OFFSET_SIGMA__uA,
        mismatch=False,
        offset=False,
        seed=20200709,
    )
    assert tuple(gated.subtractor.ratio_mismatch.shape) == (gated.n_io, 1)
    assert torch.equal(gated.subtractor.ratio_mismatch, torch.ones_like(gated.subtractor.ratio_mismatch))
    assert torch.equal(gated.subtractor.offset__uA, torch.zeros_like(gated.subtractor.offset__uA))

    # The gated chain is bit-exact against the nominal all_off build.
    nominal = _build(device)
    w_col0 = torch.ones((TINY_ROW_NUM,), dtype=torch.long)
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long)
    assert torch.equal(_probe_i_sub_col0(gated, w_col0, x), _probe_i_sub_col0(nominal, w_col0, x))


def test_offset_flips_sign_near_small_mac(device: torch.device) -> None:
    """A fabrication offset draw can flip the decoded sign of a |M| = 1 column.

    Statistical but seed-fixed: sweeping a fixed seed range, the ideal
    phase-0 ``M = +1`` column must decode NEGATIVE for some draws (the offset
    overwhelms the ~1.65 uA one-unit difference and crosses the first
    threshold on the flipped side) and stay strictly positive for others.
    """
    w_col0 = torch.zeros((TINY_ROW_NUM,), dtype=torch.long)
    w_col0[0] = 1  # single +1 row in phase 0
    x = torch.zeros((TINY_ROW_NUM,), dtype=torch.long)
    x[0] = 1  # drive exactly that row: ideal phase-0 M = +1

    codes: list[int] = []
    for seed in _SEEDS:
        xbar = _build(device, offset_sigma__uA=_OFFSET_SIGMA__uA, offset=True, seed=seed)
        assert not torch.equal(xbar.subtractor.offset__uA, torch.zeros_like(xbar.subtractor.offset__uA))
        codes.append(_decode_col0_phase0(xbar, w_col0, x))

    assert any(c < 0 for c in codes), f"no sign flip across seeds {list(_SEEDS)}: {codes}"
    assert any(c > 0 for c in codes), f"offset should not always flip: {codes}"
    assert all(-MAG_MAX <= c <= MAG_MAX for c in codes)

    # The M = 0 balance point (one +1 row and one -1 row, both driven, both in
    # phase 0) is perturbed to nonzero codes of either sign across draws.
    w_bal = torch.zeros((TINY_ROW_NUM,), dtype=torch.long)
    w_bal[0], w_bal[1] = 1, -1
    x_bal = torch.zeros((TINY_ROW_NUM,), dtype=torch.long)
    x_bal[:2] = 1
    bal_codes = [
        _decode_col0_phase0(_build(device, offset_sigma__uA=_OFFSET_SIGMA__uA, offset=True, seed=seed), w_bal, x_bal)
        for seed in _SEEDS
    ]
    assert any(c != 0 for c in bal_codes), f"offset must perturb the M=0 balance: {bal_codes}"
    assert any(c > 0 for c in bal_codes) and any(c < 0 for c in bal_codes), (
        f"M=0 perturbation should land on both signs across draws: {bal_codes}"
    )


def test_mismatch_perturbs_magnitude(device: torch.device) -> None:
    """Subtracted-leg ratio mismatch shifts I_SUB and the decoded magnitude."""
    w_col0 = -torch.ones((TINY_ROW_NUM,), dtype=torch.long)  # strong -1 column:
    x = torch.ones((TINY_ROW_NUM,), dtype=torch.long)  # the subtracted N leg dominates

    nominal = _build(device)
    i_sub_nominal = _probe_i_sub_col0(nominal, w_col0, x)
    code_nominal = _decode_col0_phase0(nominal, w_col0, x)
    assert code_nominal == -MAG_MAX  # ideal per-phase M = -8 saturates at -7

    perturbed_codes: list[int] = []
    for seed in _SEEDS:
        on = _build(device, mismatch_sigma=_MISMATCH_SIGMA, mismatch=True, seed=seed)
        assert not torch.equal(on.subtractor.ratio_mismatch, torch.ones_like(on.subtractor.ratio_mismatch))

        # Deterministic analog assertion: the magnitude path changes.
        i_sub_on = _probe_i_sub_col0(on, w_col0, x)
        assert not torch.equal(i_sub_on, i_sub_nominal)

        # Eval-mode repeatability on one fabricate (no per-call draw).
        assert torch.equal(i_sub_on, _probe_i_sub_col0(on, w_col0, x))

        perturbed_codes.append(_decode_col0_phase0(on, w_col0, x))

    # A ~30% relative error on the ~12 uA dominant leg moves the decoded
    # magnitude out of the ideal -7 bin for some fabrication draws.
    assert any(c != code_nominal for c in perturbed_codes), (
        f"mismatch never moved the decoded magnitude: {perturbed_codes}"
    )
    # The sign stays negative: the ratio scales the dominant leg, it does not
    # change which leg dominates at this saturating -8 phase.
    assert all(c <= 0 for c in perturbed_codes), f"unexpected sign flip: {perturbed_codes}"

"""CurrentSubtractor: magnitude-and-sign current-difference primitive.

With both nonidealities off the subtractor is the exact reference transform
``i_diff = gain·|i_a - i_b|`` with ``sign = (i_a < i_b)``; enabling the
subtracted-leg ratio mismatch or the sign-comparator offset perturbs the
output away from that reference. Config validation rejects a non-positive
gain and a negative sigma.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.analog.current_subtractor import (
    CurrentSubtractor,
    CurrentSubtractorConfig,
    CurrentSubtractorPolicy,
)


def _make(
    *,
    gain: float = 2.0,
    mismatch_sigma_relative: float = 0.1,
    offset_sigma__uA: float = 0.5,
    mismatch: bool = False,
    offset: bool = False,
    inst_shape: tuple[int, ...] = (),
) -> CurrentSubtractor:
    sub = CurrentSubtractor(
        config=CurrentSubtractorConfig(
            gain=gain,
            mismatch_sigma_relative=mismatch_sigma_relative,
            offset_sigma__uA=offset_sigma__uA,
        ),
        policy=CurrentSubtractorPolicy(mismatch=mismatch, offset=offset),
        inst_shape=inst_shape,
        dtype=torch.float64,
        T__K=300.0,
    )
    sub.eval()
    return sub


def test_all_off_is_exact_magnitude_and_sign() -> None:
    """Both toggles off: exact ``gain·|a - b|`` magnitude and ``(a < b)`` sign."""
    gain = 2.0
    sub = _make(gain=gain)

    i_a__uA = torch.tensor([10.0, -4.0, 3.0, 5.0], dtype=torch.float64)
    i_b__uA = torch.tensor([4.0, -4.0, 7.0, -2.0], dtype=torch.float64)

    i_diff__uA, sign = sub.subtract(i_a__uA, i_b__uA)

    torch.testing.assert_close(i_diff__uA, gain * (i_a__uA - i_b__uA).abs())
    assert torch.equal(sign, i_a__uA < i_b__uA)
    # The magnitude is unipolar; the direction travels in the sign bit.
    assert (i_diff__uA >= 0).all()


def test_mismatch_scales_subtracted_leg() -> None:
    """With ``mismatch`` on the subtracted leg carries the held per-instance ratio."""
    sub = _make(mismatch=True, inst_shape=(4,))
    torch.manual_seed(0)
    sub._sample_fabricate_mismatch()
    assert not torch.allclose(sub.ratio_mismatch, torch.ones_like(sub.ratio_mismatch))

    i_a__uA = torch.tensor([10.0, -4.0, 3.0, 5.0], dtype=torch.float64)
    i_b__uA = torch.tensor([4.0, -4.0, 7.0, -2.0], dtype=torch.float64)

    i_diff__uA, _ = sub.subtract(i_a__uA, i_b__uA)
    torch.testing.assert_close(i_diff__uA, 2.0 * (i_a__uA - sub.ratio_mismatch * i_b__uA).abs())
    # Departs from the exact-difference reference.
    assert not torch.allclose(i_diff__uA, 2.0 * (i_a__uA - i_b__uA).abs())


def test_offset_shifts_difference_and_sets_sign_at_balance() -> None:
    """With ``offset`` on the input-referred offset shifts the difference and drives the sign at balance."""
    sub = _make(offset=True, inst_shape=(4,))
    torch.manual_seed(0)
    sub._sample_fabricate_mismatch()
    assert not torch.allclose(sub.offset__uA, torch.zeros_like(sub.offset__uA))

    # Balanced legs: delta is exactly the held offset (ratio unit, mismatch off).
    i_bal__uA = torch.tensor([5.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    i_diff__uA, sign = sub.subtract(i_bal__uA, i_bal__uA)

    torch.testing.assert_close(i_diff__uA, 2.0 * sub.offset__uA.abs())
    assert torch.equal(sign, sub.offset__uA < 0)


def test_validation_rejects_bad_config() -> None:
    """A non-positive gain and a negative sigma are rejected; zero sigmas pass."""
    base = {"gain": 2.0, "mismatch_sigma_relative": 0.1, "offset_sigma__uA": 0.5}
    for override in (
        {"gain": 0.0},
        {"gain": -1.0},
        {"mismatch_sigma_relative": -1e-3},
        {"offset_sigma__uA": -1e-3},
    ):
        with pytest.raises(ValueError):
            CurrentSubtractorConfig(**{**base, **override})

    # Zero sigmas denote an ideal (mismatch-free, offset-free) subtractor.
    CurrentSubtractorConfig(gain=1.0, mismatch_sigma_relative=0.0, offset_sigma__uA=0.0)

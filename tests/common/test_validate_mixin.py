"""Tests for the reusable validation predicates."""

from __future__ import annotations

import pytest

from neurox.common import ValidateMixin


@pytest.mark.parametrize(
    ("method", "value", "ref"),
    [
        (ValidateMixin._require_gt, float("inf"), 0.0),
        (ValidateMixin._require_ge, float("inf"), 0.0),
        (ValidateMixin._require_lt, float("-inf"), 0.0),
        (ValidateMixin._require_le, float("-inf"), 0.0),
        (ValidateMixin._require_gt, 1.0, float("-inf")),
        (ValidateMixin._require_le, 1.0, float("inf")),
    ],
)
def test_comparison_helpers_reject_non_finite_operands(method, value: float, ref: float) -> None:
    with pytest.raises(ValueError, match="both operands must be finite"):
        method(value, "value", ref)


def test_scalar_helpers_reject_non_finite_values() -> None:
    for method in (ValidateMixin._require_finite, ValidateMixin._require_pos, ValidateMixin._require_non_neg):
        with pytest.raises(ValueError, match="finite"):
            method(float("inf"), "value")


def test_closed_interval_helper_includes_both_finite_bounds() -> None:
    ValidateMixin._require_in_closed_interval(0.0, "value", 0.0, 1.0)
    ValidateMixin._require_in_closed_interval(1.0, "value", 0.0, 1.0)
    for value in (-0.1, 1.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match=r"in closed interval \[0\.0, 1\.0\]"):
            ValidateMixin._require_in_closed_interval(value, "value", 0.0, 1.0)


def test_order_helpers_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        ValidateMixin._require_increasing((0.0, float("inf")), "values")
    with pytest.raises(ValueError, match="finite"):
        ValidateMixin._require_decreasing((0.0, float("-inf")), "values")


def test_length_helpers() -> None:
    ValidateMixin._require_non_empty((1,), "values")
    ValidateMixin._require_len((1, 2), "values", 2)
    ValidateMixin._require_same_len((1,), "left", (2,), "right")
    ValidateMixin._require_min_len((1, 2), "values", 1)

    with pytest.raises(ValueError, match=r"len\(values\) \(0\) >= 1"):
        ValidateMixin._require_non_empty((), "values")
    with pytest.raises(ValueError, match=r"len\(values\) \(1\) == 2"):
        ValidateMixin._require_len((1,), "values", 2)
    with pytest.raises(ValueError, match=r"len\(left\) \(1\) == len\(right\) \(2\)"):
        ValidateMixin._require_same_len((1,), "left", (2, 3), "right")
    with pytest.raises(ValueError, match=r"len\(values\) \(1\) >= 2"):
        ValidateMixin._require_min_len((1,), "values", 2)

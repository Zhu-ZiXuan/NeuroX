"""Named sample layouts and working windows stay aligned through reporting."""

from __future__ import annotations

import torch

from neurox import ProfileItem, Reporter


def test_static_energy_uses_the_nearest_working_window_at_each_sample() -> None:
    first = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    second = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4) + 100
    result = {
        "unit": ProfileItem(
            area__um2=6.0,
            leakage__uW=1.5,
            dynamic_energy__fJ=None,
            working_duration__ns=torch.tensor([5.0, 5.0, 5.0, 7.0], dtype=torch.float64)[None, :, None].expand(2, 4, 4),
        ),
        "unit.adc": ProfileItem(
            area__um2=18.0,
            leakage__uW=1.5,
            dynamic_energy__fJ=torch.cat((first, second), dim=1),
            working_duration__ns=None,
        ),
        "unit.clock": ProfileItem(
            area__um2=3.0,
            leakage__uW=0.375,
            dynamic_energy__fJ=None,
            working_duration__ns=torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float64)[None, :, None].expand(2, 4, 4),
        ),
    }
    reporter = Reporter(result)
    expected_energy = torch.cat((first, second), dim=1)
    expected_duration = torch.tensor([5.0, 5.0, 5.0, 7.0], dtype=torch.float64)[None, :, None].expand(2, 4, 4)
    expected_clock = torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float64)[None, :, None].expand(2, 4, 4)

    assert set(reporter.data) == set(result)
    assert reporter.breakdown("area") == {"unit": 6.0, "unit.adc": 18.0, "unit.clock": 3.0}
    torch.testing.assert_close(reporter.data["unit.adc"].dynamic_energy__fJ, expected_energy, check_dtype=False)
    torch.testing.assert_close(reporter.data["unit.adc"].working_duration__ns, expected_duration)
    torch.testing.assert_close(reporter.data["unit"].static_energy__fJ, 1.5 * expected_duration, check_dtype=False)
    torch.testing.assert_close(reporter.data["unit.adc"].static_energy__fJ, 1.5 * expected_duration, check_dtype=False)
    torch.testing.assert_close(reporter.data["unit.clock"].static_energy__fJ, 0.375 * expected_clock, check_dtype=False)
    torch.testing.assert_close(
        reporter.breakdown("total_energy")["unit.adc"],
        expected_energy + 1.5 * expected_duration,
        check_dtype=False,
    )
    assert result["unit.adc"].working_duration__ns is None
    torch.testing.assert_close(reporter.data["unit.adc"].powered_duration__ns, expected_duration)

    powered = expected_duration + 10.0
    overridden = Reporter(result, powered_duration__ns={"unit": powered, "unit.adc": torch.zeros_like(powered)})
    torch.testing.assert_close(overridden.data["unit"].static_energy__fJ, 1.5 * powered, check_dtype=False)
    torch.testing.assert_close(
        overridden.data["unit.adc"].static_energy__fJ, torch.zeros_like(powered), check_dtype=False
    )
    torch.testing.assert_close(overridden.data["unit.adc"].working_duration__ns, expected_duration)
    # Exact-name replacements leave the child's own default window intact.
    torch.testing.assert_close(overridden.data["unit.clock"].powered_duration__ns, expected_clock)
    torch.testing.assert_close(overridden.breakdown("powered_duration")["unit"], powered)
    torch.testing.assert_close(result["unit"].working_duration__ns, expected_duration)


def test_missing_costs_remain_unknown_and_virtual_channels_gain_no_hardware() -> None:
    result = {
        "unit": ProfileItem(
            area__um2=0.0,
            leakage__uW=0.0,
            dynamic_energy__fJ=None,
            working_duration__ns=torch.zeros(2, 3, dtype=torch.float64),
        ),
        "unit.channel": ProfileItem(
            area__um2=None,
            leakage__uW=None,
            dynamic_energy__fJ=torch.ones(2, 3),
            working_duration__ns=None,
        ),
        "untimed": ProfileItem(
            area__um2=4.0,
            leakage__uW=2.0,
            dynamic_energy__fJ=None,
            working_duration__ns=None,
        ),
    }
    reporter = Reporter(result)
    torch.testing.assert_close(
        reporter.data["unit"].static_energy__fJ, torch.zeros(2, 3, dtype=torch.float64), check_dtype=False
    )
    assert reporter.data["unit.channel"].static_energy__fJ is None
    assert reporter.data["unit.channel"].area__um2 is None
    assert reporter.data["untimed"].area__um2 == 4.0
    assert reporter.data["untimed"].static_energy__fJ is None
    overridden = Reporter(result, powered_duration__ns={"untimed": torch.full((2, 3), 3.0)})
    torch.testing.assert_close(overridden.data["untimed"].static_energy__fJ, torch.full((2, 3), 6.0), check_dtype=False)
    assert overridden.data["untimed"].working_duration__ns is None
    assert result["untimed"].working_duration__ns is None
    torch.testing.assert_close(reporter.breakdown("total_energy")["unit.channel"], torch.ones(2, 3), check_dtype=False)

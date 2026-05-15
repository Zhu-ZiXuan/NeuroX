"""Shared pytest fixtures for ``packages/<sub>/tests``.

Replicated across every sub-package so each ``pytest packages/<sub>/tests``
invocation discovers the same ``--device`` CLI option and ``device``
fixture.  When multiple per-package test dirs are unioned in a single
pytest run, the duplicate ``--device`` registration is silently ignored
by the ``ValueError`` guard — only the first conftest to load wins, all
others reuse the same option.
"""

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            help="Target torch device for compatibility tests, e.g. cpu, cuda, cuda:0, mps.",
        )
    except ValueError:
        # Another per-package conftest in this same pytest session already
        # registered ``--device``; reuse the existing option.
        pass


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    device_name = str(request.config.getoption("device"))
    target = torch.device(device_name)

    try:
        torch.empty(0, device=target)
    except (RuntimeError, AssertionError) as exc:
        pytest.skip(f"Device {device_name} is not available: {exc}")

    return target

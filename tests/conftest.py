"""Shared pytest fixtures for ``packages/<sub>/tests``.

Replicated across every sub-package so each ``pytest packages/<sub>/tests``
invocation discovers the same ``--device`` CLI option and ``device``
fixture.  When multiple per-package test dirs are unioned in a single
pytest run, the duplicate ``--device`` registration is silently ignored
by the ``ValueError`` guard — only the first conftest to load wins, all
others reuse the same option.
"""

import contextlib

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    # Another per-package conftest in this same pytest session may
    # have already registered ``--device``; suppress the duplicate
    # registration ValueError and reuse the existing option.
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            help="Target torch device for compatibility tests, e.g. cpu, cuda, cuda:0, mps.",
        )


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    device_name = str(request.config.getoption("device"))
    target = torch.device(device_name)

    try:
        torch.empty(0, device=target)
    except (RuntimeError, AssertionError) as exc:
        pytest.skip(f"Device {device_name} is not available: {exc}")

    return target

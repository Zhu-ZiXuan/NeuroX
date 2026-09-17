"""Caller-selected device placement and isolation between compiled model fixtures."""

import contextlib

import pytest
import torch


@pytest.fixture(autouse=True)
def isolate_compiler_cache() -> None:
    # Independent models specialize the same functions; variants from earlier
    # tests must not consume the next test's compiler recompile budget.
    torch.compiler.reset()


def pytest_addoption(parser: pytest.Parser) -> None:
    # A sibling conftest may already have registered `--device`; reuse it.
    with contextlib.suppress(ValueError):
        parser.addoption(
            "--device",
            action="store",
            default="cpu",
            help="Target torch device for compatibility tests, e.g. cpu, cuda, cuda:0, mps.",
        )


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> torch.device:
    device_name = str(request.config.getoption("device"))
    target = torch.device(device_name)

    try:
        torch.empty(0, device=target)
    except (RuntimeError, AssertionError) as exc:
        pytest.skip(f"Device {device_name} is not available: {exc}")

    return target

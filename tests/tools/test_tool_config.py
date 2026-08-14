"""Tests for the shared tool CLI helper `neurox.tools._config`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from neurox.tools._config import add_standard_args, resolve_relative_path, setup_logging


class TestAddStandardArgs:
    def test_device_default_is_cpu(self) -> None:
        """Omitting `--device` must resolve to a concrete CPU device."""
        parser = argparse.ArgumentParser()
        add_standard_args(parser)
        args = parser.parse_args(["--config", "/tmp/x.toml"])
        assert args.device == "cpu"
        assert torch.device(args.device) == torch.device("cpu")

    def test_explicit_device_is_passed_through(self) -> None:
        parser = argparse.ArgumentParser()
        add_standard_args(parser)
        args = parser.parse_args(["--config", "/tmp/x.toml", "--device", "cuda:0"])
        assert args.device == "cuda:0"

    def test_device_can_be_suppressed(self) -> None:
        """`device=False` removes `--device` for CPU-only tools."""
        parser = argparse.ArgumentParser()
        add_standard_args(parser, device=False)
        args = parser.parse_args(["--config", "/tmp/x.toml"])
        assert not hasattr(args, "device")

    def test_log_level_rejects_unknown_value(self) -> None:
        """Argparse choices must catch typos at parse time."""
        parser = argparse.ArgumentParser()
        add_standard_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "/tmp/x.toml", "--log-level", "VERBOSE"])

    def test_log_level_case_insensitive(self) -> None:
        parser = argparse.ArgumentParser()
        add_standard_args(parser)
        args = parser.parse_args(["--config", "/tmp/x.toml", "--log-level", "debug"])
        assert args.log_level == "DEBUG"


class TestSetupLogging:
    def test_accepts_known_levels(self) -> None:
        """Smoke: every documented level is accepted without raising."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            setup_logging(level)

    def test_rejects_unknown_level(self) -> None:
        with pytest.raises(AttributeError):
            setup_logging("NOSUCHLEVEL")


class TestResolveRelativePath:
    def test_relative_resolves_against_base_parent(self, tmp_path: Path) -> None:
        base = tmp_path / "subdir" / "main.toml"
        base.parent.mkdir()
        out = resolve_relative_path("artefacts/out.png", base)
        assert out == tmp_path / "subdir" / "artefacts/out.png"

    def test_absolute_returned_unchanged(self, tmp_path: Path) -> None:
        absolute = tmp_path / "fixed.png"
        out = resolve_relative_path(absolute, tmp_path / "main.toml")
        assert out == absolute

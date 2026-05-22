"""Tests for the staged ``build_evaluator`` orchestration.

Contract:

* ``replace_model`` + ``load_neurox_state`` + ``bind_output_calibration``
  + ``program_model`` + ``fabricate_model`` produces the same end state
  as the one-shot ``build_evaluator`` wrapper;
* ``load_neurox_state`` raises ``NeuroxStateError`` when required NeuroX
  operator buffers are missing in the checkpoint (strict mode);
* ``StateBindingReport`` carries the loaded / missing / unexpected
  diagnostic for non-strict callers.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neurox.macro.xbar import IdealXbarMacro, IdealXbarMacroConfig, XbarMacro
from neurox.replace import (
    NeuroxStateError,
    StateBindingReport,
    bind_output_calibration,
    build_evaluator,
    fabricate_model,
    load_neurox_state,
    program_model,
    replace_model,
)


def _fake_macro_factory(*, name: str = "", w_logical_shape: tuple[int, ...] = (1, 1)) -> IdealXbarMacro:
    """Zero-cost macro factory respecting the name + w_logical_shape contract."""
    cfg = IdealXbarMacroConfig(x_value_range=(-7, 7), w_value_range=(-7, 7))
    macro = XbarMacro.from_config(
        cfg=cfg,
        name=name,
        w_logical_shape=w_logical_shape,
        dtype=torch.float32,
        T__K=300.0,
        ideal_xbar=False,
    )
    assert isinstance(macro, IdealXbarMacro)
    return macro


def _toy_model(in_f: int = 4, out_f: int = 3) -> nn.Module:
    """Minimal float model — a single Linear layer named ``fc``."""

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(in_f, out_f)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    return Toy()


def _build_checkpoint(model: nn.Module) -> dict[str, object]:
    """Replace ``model`` once, then snapshot the resulting buffers.

    Returns a ``neurox_flat`` checkpoint payload that fully covers the
    replaced model's required keys.  The model is left replaced.
    """
    replace_model(model, _fake_macro_factory, verbose=False)
    # The replaced QuantLinear has zero-initialised int buffers; that's
    # fine for the round-trip test because we only check schema parity.
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return {"schema": "neurox_flat", "state_dict": state}


class TestStaging:
    """The five staged calls equal the one-shot wrapper."""

    def test_orchestration_equals_wrapper(self) -> None:
        staged_model = _toy_model()
        replace_model(staged_model, _fake_macro_factory, verbose=False)
        # snapshot the replaced-model state for use as the checkpoint
        ckpt = {"schema": "neurox_flat", "state_dict": staged_model.state_dict()}
        report = load_neurox_state(staged_model, ckpt, strict=True)
        bind_output_calibration(staged_model)
        program_model(staged_model)
        fabricate_model(staged_model)

        wrapper_model = _toy_model()
        build_evaluator(wrapper_model, ckpt, _fake_macro_factory, verbose=False)

        # Same buffer values on both paths.
        for k in staged_model.state_dict():
            assert torch.equal(
                staged_model.state_dict()[k],
                wrapper_model.state_dict()[k],
            ), f"mismatch on {k}"
        # Report identifies the single replaced operator and no missing keys.
        assert report.loaded == ["fc"]
        assert report.missing == []


class TestStrictLoad:
    """Missing required buffers raise loud errors in strict mode."""

    def test_missing_keys_raise(self) -> None:
        model = _toy_model()
        ckpt = _build_checkpoint(model)
        # Drop a required key.
        ckpt["state_dict"].pop("fc.weight_int")  # type: ignore[union-attr]
        with pytest.raises(NeuroxStateError) as excinfo:
            load_neurox_state(model, ckpt, strict=True)
        assert "fc.weight_int" in excinfo.value.missing
        assert excinfo.value.unexpected == []

    def test_relaxed_load_returns_report(self) -> None:
        model = _toy_model()
        ckpt = _build_checkpoint(model)
        ckpt["state_dict"].pop("fc.bias_int")  # type: ignore[union-attr]
        report = load_neurox_state(model, ckpt, strict=False)
        assert isinstance(report, StateBindingReport)
        assert "fc.bias_int" in report.missing


class TestSchemaErrors:
    """Schema-marker validation rejects unknown formats."""

    def test_missing_schema_marker(self) -> None:
        model = _toy_model()
        replace_model(model, _fake_macro_factory, verbose=False)
        with pytest.raises(ValueError, match="unsupported checkpoint schema"):
            load_neurox_state(model, {"state_dict": {}}, strict=False)

    def test_wrong_schema_marker(self) -> None:
        model = _toy_model()
        replace_model(model, _fake_macro_factory, verbose=False)
        with pytest.raises(ValueError, match="unsupported checkpoint schema"):
            load_neurox_state(model, {"schema": "legacy_v1", "state_dict": {}}, strict=False)

    def test_state_dict_missing(self) -> None:
        model = _toy_model()
        replace_model(model, _fake_macro_factory, verbose=False)
        with pytest.raises(ValueError, match="state_dict"):
            load_neurox_state(model, {"schema": "neurox_flat"}, strict=False)

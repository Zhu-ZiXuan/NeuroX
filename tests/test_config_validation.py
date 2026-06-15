"""Config-validation regression tests.

Covers the rules tightened across R6:
- ``w_state_offset < len(state_to_g_map)`` (so digit 0 maps to a state).
- ``Offset1T1RXbarConfig.adc_calibration`` rejects duplicate
  ``(adc_mode, adc_bits)`` and non-positive ``rescale_factor``.
- ``IdealXbar.adc_rescale_factor`` raises ``KeyError`` when ``adc_bits``
  exceeds ``adc_max_bits`` (lookup is bit-width-keyed; ``adc_mode`` is
  not consulted).
- ``IdealXbar.program`` rejects floating-point digit tensors.
- ``load_dump`` rejects primitive type mis-coercion, rejects ``str`` for
  list/tuple/set fields, round-trips polymorphic dataclasses via
  ``_neurox_type``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from neurox.analog.adc import (
    AdcCalibrationRecord,
    AdcOperationPoint,
)
from neurox.common import dataclass_from_dict, dataclass_from_file, dataclass_to_dict
from neurox.xbar import Offset1T1RXbarConfig
from neurox.xbar.ideal import IdealXbar, IdealXbarConfig, IdealXbarPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET = REPO_ROOT / "example" / "config" / "1t1r_28nm.toml"


# ---------------------------------------------------------------------------
# w_state_offset upper bound
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_cfg() -> Offset1T1RXbarConfig:
    if not PRESET.is_file():
        pytest.skip(f"missing chip preset: {PRESET}")
    return dataclass_from_file(Offset1T1RXbarConfig, PRESET, section="xbar")


class TestOffsetVsStateMap:
    def test_offset_at_state_count_rejected(self, base_cfg: Offset1T1RXbarConfig) -> None:
        from dataclasses import replace

        n_states = len(base_cfg.core_config.state_to_g_map__uS)
        with pytest.raises(ValueError, match=r"w_state_offset.*<.*len\(state_to_g_map"):
            replace(base_cfg, w_state_offset=n_states)

    def test_offset_above_state_count_rejected(self, base_cfg: Offset1T1RXbarConfig) -> None:
        from dataclasses import replace

        n_states = len(base_cfg.core_config.state_to_g_map__uS)
        with pytest.raises(ValueError, match=r"w_state_offset.*<.*len\(state_to_g_map"):
            replace(base_cfg, w_state_offset=n_states + 5)


# ---------------------------------------------------------------------------
# adc_calibration duplicate / non-positive rescale
# ---------------------------------------------------------------------------


class TestAdcCalibrationValidate:
    def test_duplicate_mode_bits_rejected(self, base_cfg: Offset1T1RXbarConfig) -> None:
        from dataclasses import replace

        first = base_cfg.adc_calibration[0]
        dup = AdcCalibrationRecord(
            adc_mode=first.adc_mode,
            adc_bits=first.adc_bits,
            rescale_factor=first.rescale_factor * 2.0,
        )
        with pytest.raises(ValueError, match=r"duplicate"):
            replace(base_cfg, adc_calibration=(*tuple(base_cfg.adc_calibration), dup))

    def test_negative_rescale_rejected(self, base_cfg: Offset1T1RXbarConfig) -> None:
        from dataclasses import replace

        bad = AdcCalibrationRecord(adc_mode=99, adc_bits=8, rescale_factor=-1.0)
        with pytest.raises(ValueError, match=r"rescale_factor"):
            replace(base_cfg, adc_calibration=(*tuple(base_cfg.adc_calibration), bad))


# ---------------------------------------------------------------------------
# IdealXbar.adc_rescale_factor bit-width lookup
# ---------------------------------------------------------------------------


class TestIdealRescaleFactor:
    def _build_xbar(self) -> IdealXbar:
        cfg = IdealXbarConfig(
            col_num=4,
            row_num=4,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            x_range=(0, 1),
            w_digit_count=1,
            w_digit_radix=2,
            w_digit_range=(0, 1),
            adc_mode_num=1,
            adc_max_bits=8,
        )
        return IdealXbar(
            config=cfg,
            policy=IdealXbarPolicy(),
            name="x",
            inst_shape=(),
            dtype=torch.float32,
            T__K=300.0,
        )

    def test_bits_above_max_raises(self) -> None:
        xbar = self._build_xbar()
        with pytest.raises(KeyError):
            xbar.adc_rescale_factor(AdcOperationPoint(adc_mode=0, adc_bits=9))

    def test_mode_is_ignored(self) -> None:
        """Any ``adc_mode`` returns the same rescale for a given ``adc_bits``."""
        xbar = self._build_xbar()
        r0 = xbar.adc_rescale_factor(AdcOperationPoint(adc_mode=0, adc_bits=4))
        r999 = xbar.adc_rescale_factor(AdcOperationPoint(adc_mode=999, adc_bits=4))
        assert r0 == r999


# ---------------------------------------------------------------------------
# IdealXbar.program dtype guard
# ---------------------------------------------------------------------------


class TestIdealProgramDtype:
    def test_float_dtype_rejected(self) -> None:
        cfg = IdealXbarConfig(
            col_num=4,
            row_num=4,
            area_per_inst__um2=0.0,
            leakage_per_inst__uW=0.0,
            x_range=(0, 1),
            w_digit_count=1,
            w_digit_radix=2,
            w_digit_range=(0, 1),
            adc_mode_num=1,
            adc_max_bits=8,
        )
        xbar = IdealXbar(
            config=cfg,
            policy=IdealXbarPolicy(),
            name="x",
            inst_shape=(),
            dtype=torch.float32,
            T__K=300.0,
        )
        bad = torch.zeros(xbar._w_layout_shape, dtype=torch.float32)
        with pytest.raises(TypeError, match=r"integer digit tensor"):
            xbar.program(bad)


# ---------------------------------------------------------------------------
# load_dump primitive/list/Polymorphic round-trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scalars:
    n: int
    f: float
    flag: bool
    name: str


@dataclass(frozen=True)
class _WithList:
    xs: list[int]


class TestPrimitiveGuards:
    def test_int_rejects_string(self) -> None:
        with pytest.raises(TypeError, match=r"Expected int"):
            dataclass_from_dict(_Scalars, {"n": "1", "f": 1.0, "flag": True, "name": "a"})

    def test_int_rejects_bool(self) -> None:
        with pytest.raises(TypeError, match=r"Expected int"):
            dataclass_from_dict(_Scalars, {"n": True, "f": 1.0, "flag": True, "name": "a"})

    def test_float_accepts_int(self) -> None:
        obj = dataclass_from_dict(_Scalars, {"n": 1, "f": 1, "flag": True, "name": "a"})
        assert obj.f == 1.0

    def test_bool_rejects_int(self) -> None:
        with pytest.raises(TypeError, match=r"Expected bool"):
            dataclass_from_dict(_Scalars, {"n": 1, "f": 1.0, "flag": 1, "name": "a"})

    def test_str_rejects_int(self) -> None:
        with pytest.raises(TypeError, match=r"Expected str"):
            dataclass_from_dict(_Scalars, {"n": 1, "f": 1.0, "flag": True, "name": 7})


class TestListRejectsString:
    def test_string_rejected_for_list(self) -> None:
        with pytest.raises(TypeError, match=r"Expected list/tuple/set"):
            dataclass_from_dict(_WithList, {"xs": "12"})


# ---------------------------------------------------------------------------
# Polymorphic round-trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PolyBase:
    base_field: int = 0


@dataclass(frozen=True)
class _PolyChildA(_PolyBase):
    a_only: int = 1


@dataclass(frozen=True)
class _PolyChildB(_PolyBase):
    b_only: str = "hi"


@dataclass(frozen=True)
class _PolyOuter:
    payload: _PolyBase


class TestPolymorphicRoundTrip:
    def test_round_trip_preserves_subclass(self) -> None:
        obj = _PolyOuter(payload=_PolyChildA(base_field=3, a_only=42))
        dumped = dataclass_to_dict(obj)
        assert dumped["payload"]["_neurox_type"] == "_PolyChildA"
        rebuilt = dataclass_from_dict(_PolyOuter, dumped)
        assert isinstance(rebuilt.payload, _PolyChildA)
        assert rebuilt.payload.base_field == 3
        assert rebuilt.payload.a_only == 42

    def test_round_trip_picks_correct_sibling(self) -> None:
        obj = _PolyOuter(payload=_PolyChildB(base_field=9, b_only="x"))
        dumped = dataclass_to_dict(obj)
        rebuilt = dataclass_from_dict(_PolyOuter, dumped)
        assert isinstance(rebuilt.payload, _PolyChildB)
        assert rebuilt.payload.b_only == "x"

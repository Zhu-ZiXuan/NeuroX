"""Quantization surface of the CimMacro base.

- `validate_quantization_input_range`: only the two canonical window
  shapes (unsigned `[0, U]`, mid-zero `[-m, m - 1]`) are legal.
- `map_magnitude_input_code` / `map_zero_point_input_code`: the two ADC
  input-code grids a macro can discriminate on, each with its inclusive range.
- `CimMacroMode`: the per-mode metadata record of a physical macro.
- concrete `rescale_factor` / `to_ideal`: the uniform bit-width law and
  the ideal twin built from the published windows.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroMode,
    CimMacroPolicy,
    IdealCimMacro,
    map_magnitude_input_code,
    map_zero_point_input_code,
    validate_quantization_input_range,
)

_CANONICAL = [(0, 1), (0, 7), (0, 223), (-1, 0), (-8, 7), (-6, 5), (-128, 127)]
_NON_CANONICAL = [(-8, 8), (-1, 1), (-128, 128), (0, 0), (0, -1), (1, 5), (-5, -1), (-4, 6), (-4, 2)]


class TestQuantizationInputRangeValidator:
    """Canonical windows put the zero point on a bin edge at every width."""

    @pytest.mark.parametrize("window", _CANONICAL)
    def test_canonical_accepted(self, window: tuple[int, int]) -> None:
        validate_quantization_input_range(window)

    @pytest.mark.parametrize("window", _NON_CANONICAL)
    def test_non_canonical_rejected(self, window: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            validate_quantization_input_range(window)

    def test_symmetric_window_rejected(self) -> None:
        """`[-n, n]` holds an odd level count: zero sits inside a bin."""
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            validate_quantization_input_range((-32, 32))

    def test_mid_zero_width_need_not_be_a_power_of_two(self) -> None:
        for m in (3, 5, 6, 7, 14, 223):
            validate_quantization_input_range((-m, m - 1))

    @pytest.mark.parametrize("window", _CANONICAL)
    def test_zero_lands_on_a_bin_edge_at_every_width(self, window: tuple[int, int]) -> None:
        """The shape rule is exactly `-lower / W` in `{0, 1/2}`."""
        lower, upper = window
        width = upper - lower + 1
        assert -lower * 2 % width == 0
        assert -lower / width in (0.0, 0.5)


class TestInputCodeMaps:
    """Each map returns the mapped codes and their inclusive range."""

    def test_magnitude_map_folds_the_sign(self) -> None:
        code = torch.tensor([-8, -3, 0, 3, 7], dtype=torch.int64)
        mapped, code_range = map_magnitude_input_code(code, code_range=(-8, 7))
        assert torch.equal(mapped, torch.tensor([8, 3, 0, 3, 7], dtype=torch.int64))
        assert code_range == (0, 7)

    def test_magnitude_map_leaves_the_phantom_outside_the_grid(self) -> None:
        """A mid-zero window's bottom value has no magnitude tap."""
        mapped, (_, upper) = map_magnitude_input_code(torch.tensor([-8]), code_range=(-8, 7))
        assert int(mapped.item()) == upper + 1

    def test_magnitude_map_is_identity_on_an_unsigned_window(self) -> None:
        code = torch.tensor([0, 5, 15], dtype=torch.int64)
        mapped, code_range = map_magnitude_input_code(code, code_range=(0, 15))
        assert torch.equal(mapped, code)
        assert code_range == (0, 15)

    def test_zero_point_map_offsets_by_the_window_floor(self) -> None:
        code = torch.tensor([-8, -1, 0, 7], dtype=torch.int64)
        mapped, code_range = map_zero_point_input_code(code, code_range=(-8, 7))
        assert torch.equal(mapped, torch.tensor([0, 7, 8, 15], dtype=torch.int64))
        assert code_range == (0, 15)

    def test_zero_point_map_is_identity_on_an_unsigned_window(self) -> None:
        code = torch.tensor([0, 100, 223], dtype=torch.int64)
        mapped, code_range = map_zero_point_input_code(code, code_range=(0, 223))
        assert torch.equal(mapped, code)
        assert code_range == (0, 223)

    def test_zero_point_range_holds_every_window_target(self) -> None:
        for lower, upper in _CANONICAL:
            _, (lo, hi) = map_zero_point_input_code(torch.zeros(1, dtype=torch.int64), code_range=(lower, upper))
            assert (lo, hi) == (0, upper - lower)

    @pytest.mark.parametrize("window", [(-8, 8), (1, 5)])
    def test_maps_reject_a_non_canonical_window(self, window: tuple[int, int]) -> None:
        code = torch.zeros(1, dtype=torch.int64)
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            map_magnitude_input_code(code, code_range=window)
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            map_zero_point_input_code(code, code_range=window)


class TestCimMacroMode:
    """Per-mode metadata of a physical macro."""

    def test_accepts_a_canonical_record(self) -> None:
        mode = CimMacroMode(
            quantization_input_range=(-8, 7),
            adc_input_code_range=(0, 7),
            max_bits_rescale_factor=1.0,
        )
        assert mode.quantization_input_range == (-8, 7)

    def test_rejects_a_non_canonical_window(self) -> None:
        with pytest.raises(ValueError, match=r"quantization_input_range"):
            CimMacroMode(
                quantization_input_range=(-8, 8),
                adc_input_code_range=(0, 7),
                max_bits_rescale_factor=1.0,
            )

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_rejects_a_non_positive_rescale_factor(self, factor: float) -> None:
        with pytest.raises(ValueError, match=r"max_bits_rescale_factor"):
            CimMacroMode(
                quantization_input_range=(0, 223),
                adc_input_code_range=(0, 15),
                max_bits_rescale_factor=factor,
            )

    @pytest.mark.parametrize("code_range", [(0, 0), (4, 2), (-1, 7)])
    def test_rejects_a_degenerate_input_code_range(self, code_range: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match=r"adc_input_code_range"):
            CimMacroMode(
                quantization_input_range=(0, 223),
                adc_input_code_range=code_range,
                max_bits_rescale_factor=0.5,
            )


# ---------------------------------------------------------------------------
# Concrete base laws through a minimal member
# ---------------------------------------------------------------------------


class _StubMacroConfig(CimMacroConfig):
    modes: tuple[CimMacroMode, ...]
    adc_max_bits: int


class _StubMacroPolicy(CimMacroPolicy):
    pass


class _StubMacro(CimMacro[_StubMacroConfig, _StubMacroPolicy]):
    """Minimal physical-style member: metadata only, no analog model."""

    @property
    def x_value_range(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def w_value_range(self) -> tuple[int, int]:
        return (-1, 1)

    @property
    def quantization_input_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(mode.quantization_input_range for mode in self.config.modes)

    @property
    def adc_max_bits(self) -> int:
        return self.config.adc_max_bits

    def _max_bits_rescale_factor(self, quantization_mode: int) -> float:
        return self.config.modes[quantization_mode].max_bits_rescale_factor

    def map_quantization_input_code(self, code: Tensor, *, quantization_mode: int) -> tuple[Tensor, tuple[int, int]]:
        mode = self.config.modes[quantization_mode]
        mapped, _ = map_magnitude_input_code(code, code_range=mode.quantization_input_range)
        return mapped, mode.adc_input_code_range

    def program(self, w: Tensor) -> None:
        raise NotImplementedError

    def vec_mat_mul(self, x: Tensor, *, quantization_mode: int, adc_bits: int | None) -> Tensor:
        raise NotImplementedError

    def initiation_interval__ns(self, *, adc_bits: int | None) -> float:
        del adc_bits
        return 0.0


def _stub_macro(
    *,
    modes: tuple[CimMacroMode, ...],
    adc_max_bits: int,
    input_num: int = 8,
    output_num: int = 4,
    max_active_num: int = 4,
    inst_shape: tuple[int, ...] = (),
) -> _StubMacro:
    config = _StubMacroConfig(
        area_per_inst__um2=1.0,
        leakage_per_inst__uW=2.0,
        max_active_num=max_active_num,
        modes=modes,
        adc_max_bits=adc_max_bits,
    )
    return _StubMacro(
        config=config,
        policy=_StubMacroPolicy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=inst_shape,
        dtype=torch.float32,
        T__K=300.0,
    )


# Two hand-written operating points. The factor is a code-to-code currency, so
# a member whose twin resolves twice as fine at max bits declares 0.5 — never
# the MAC units a code carries.
_TWO_MODES = (
    CimMacroMode(quantization_input_range=(-8, 7), adc_input_code_range=(0, 7), max_bits_rescale_factor=1.0),
    CimMacroMode(quantization_input_range=(0, 223), adc_input_code_range=(0, 223), max_bits_rescale_factor=0.5),
)


class TestRescaleFactorLaw:
    """`r_b = r_B * 2^(B - b)` for every macro, mode by mode."""

    def test_max_bits_returns_the_hook_value(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        for mode_index, mode in enumerate(_TWO_MODES):
            factor = macro.rescale_factor(quantization_mode=mode_index, adc_bits=4)
            assert factor == mode.max_bits_rescale_factor

    def test_dropping_one_bit_doubles_the_factor(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        for mode_index in range(len(_TWO_MODES)):
            for adc_bits in range(2, 5):
                coarse = macro.rescale_factor(quantization_mode=mode_index, adc_bits=adc_bits - 1)
                fine = macro.rescale_factor(quantization_mode=mode_index, adc_bits=adc_bits)
                assert coarse == 2.0 * fine

    def test_chain_is_the_max_bits_anchor_scaled_by_the_width_gap(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        for mode_index, mode in enumerate(_TWO_MODES):
            for adc_bits in range(1, 5):
                expected = mode.max_bits_rescale_factor * 2 ** (4 - adc_bits)
                assert macro.rescale_factor(quantization_mode=mode_index, adc_bits=adc_bits) == expected

    def test_lossless_oracle_is_outside_the_chain(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        assert macro.rescale_factor(quantization_mode=1, adc_bits=None) == 1.0

    @pytest.mark.parametrize("adc_bits", [0, -1, 5])
    def test_rejects_bits_outside_the_declared_width(self, adc_bits: int) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        with pytest.raises(ValueError, match=r"adc_bits"):
            macro.rescale_factor(quantization_mode=0, adc_bits=adc_bits)


class TestToIdealFromPublishedRanges:
    """The base twin is built from the published windows and geometry."""

    def test_twin_publishes_the_same_quantization_surface(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4)
        twin = macro.to_ideal()
        assert isinstance(twin, IdealCimMacro)
        assert twin.quantization_input_ranges == macro.quantization_input_ranges
        assert twin.adc_max_bits == macro.adc_max_bits
        assert twin.x_value_range == macro.x_value_range
        assert twin.w_value_range == macro.w_value_range

    def test_twin_inherits_the_logical_geometry(self) -> None:
        macro = _stub_macro(
            modes=_TWO_MODES,
            adc_max_bits=4,
            input_num=16,
            output_num=6,
            max_active_num=8,
            inst_shape=(2, 3),
        )
        twin = macro.to_ideal()
        assert (twin.input_num, twin.output_num) == (16, 6)
        assert twin.inst_shape == (2, 3)
        assert twin.max_active_num == macro.max_active_num

    def test_twin_is_the_rescale_reference(self) -> None:
        """The ideal macro anchors the currency: its own factor is 1 at max bits."""
        twin = _stub_macro(modes=_TWO_MODES, adc_max_bits=4).to_ideal()
        assert twin.rescale_factor(quantization_mode=1, adc_bits=twin.adc_max_bits) == 1.0

    def test_ideal_macro_returns_itself(self) -> None:
        twin = _stub_macro(modes=_TWO_MODES, adc_max_bits=4).to_ideal()
        assert twin.to_ideal() is twin

    def test_twin_quantizes_over_the_published_window(self) -> None:
        macro = _stub_macro(modes=_TWO_MODES, adc_max_bits=4, input_num=1, output_num=1, max_active_num=1)
        twin = macro.to_ideal()
        twin.eval()
        twin.program(torch.ones((1, 1), dtype=torch.int32))
        # Mode 0 window [-8, 7] at 4 bits: one code per MAC unit.
        dots = torch.tensor([[-9], [-8], [0], [7], [8]], dtype=torch.int64)
        codes = twin.vec_mat_mul(dots, quantization_mode=0, adc_bits=4)
        assert codes.squeeze(-1).tolist() == [-8, -8, 0, 7, 7]

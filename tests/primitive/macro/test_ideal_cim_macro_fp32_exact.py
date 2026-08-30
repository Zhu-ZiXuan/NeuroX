"""IdealCimMacro fp32-exact dot fast path.

The plane dot computation switches to fp32 einsum when
`_max_plane_dot_abs < 2^24` (every per-cell product and partial sum then
accumulates exactly in IEEE fp32) and stays on the int64 elementwise path
otherwise. Both paths must be bit-identical, on CPU and GPU, for the
exact oracle and the quantized per-plane path alike. Planes arrive
pre-masked from the caller (at most `max_active_num` selected positions each);
the macro output keeps the leading order.
"""

from __future__ import annotations

import pytest
import torch

from neurox.primitive.macro.cim import (
    CimMacroQuantizationScheme,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)


def _make_macro(
    *,
    input_num: int,
    max_active_num: int,
    output_num: int,
    x_value_range: tuple[int, int],
    w_value_range: tuple[int, int] = (-3, 3),
    rescale_factors: tuple[float, ...] = (1.0,),
    adc_bits: int = 6,
) -> IdealCimMacro:
    config = IdealCimMacroConfig(
        rescale_factors=rescale_factors,
        max_active_num=max_active_num,
        area_per_inst__um2=0.0,
        leakage_per_inst__uW=0.0,
        x_value_range=x_value_range,
        w_value_range=w_value_range,
        adc_bits=adc_bits,
        quantization_scheme=CimMacroQuantizationScheme.ZERO_POINT,
    )
    macro = IdealCimMacro(
        config=config,
        policy=IdealCimMacroPolicy(),
        input_num=input_num,
        output_num=output_num,
        inst_shape=(),
        dtype=torch.float32,
        T__K=300.0,
    )
    macro.eval()
    return macro


def _random_operands(
    macro: IdealCimMacro,
    *,
    batch: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate and program random in-range weights and inputs."""
    generator = torch.Generator().manual_seed(seed)
    w_lo, w_hi = macro.w_value_range
    w = torch.randint(
        w_lo,
        w_hi + 1,
        (macro.input_num, macro.output_num),
        dtype=torch.int32,
        generator=generator,
    )
    x_lo, x_hi = macro.x_value_range
    x = torch.randint(x_lo, x_hi + 1, (batch, macro.input_num), dtype=torch.int32, generator=generator)
    macro.program(w.to(device))
    return w, x


def _masked_planes(x: torch.Tensor, *, input_num: int, max_active_num: int) -> torch.Tensor:
    """Pre-masked WL planes via the engine mask formula."""
    p_num = input_num // max_active_num
    mask = torch.arange(input_num) // max_active_num == torch.arange(p_num).unsqueeze(-1)
    # Shape: [..., input_num] -> [..., P, input_num]
    return torch.where(mask, x.unsqueeze(-2), x.new_zeros(()))


def _plane_dot_oracle(macro: IdealCimMacro, planes: torch.Tensor) -> torch.Tensor:
    """CPU int64 exact plane dots."""
    w = macro._w.to("cpu", torch.int64)
    return (planes.to("cpu", torch.int64).unsqueeze(-1) * w).sum(dim=-2)


_REPRESENTATIVE = [
    pytest.param((0, 1), (-3, 3), id="spike_x_signed_w"),
    pytest.param((-8, 7), (0, 15), id="signed_x_unsigned_w"),
    pytest.param((0, 255), (-255, 255), id="wide_x_wide_w"),
]


class TestFastPathExact:
    """At zero active bits, the fp32 fast path equals the int64 oracle on CPU and GPU."""

    @pytest.mark.parametrize(
        ("x_value_range", "w_value_range"),
        _REPRESENTATIVE,
    )
    def test_cpu_matches_int64_oracle(
        self,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
    ) -> None:
        macro = _make_macro(
            input_num=64,
            max_active_num=16,
            output_num=8,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
        )
        assert macro._fp32_exact is True
        _, x = _random_operands(macro, batch=5, seed=101, device=torch.device("cpu"))
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        y = macro.vec_mat_mul(planes, quantization_mode=0, adc_active_bits=0)
        assert y.dtype == torch.int64
        assert y.shape == (5, 4, 8)
        assert torch.equal(y, _plane_dot_oracle(macro, planes))

    @pytest.mark.parametrize(
        ("x_value_range", "w_value_range"),
        _REPRESENTATIVE,
    )
    def test_gpu_matches_cpu_oracle(
        self,
        device: torch.device,
        x_value_range: tuple[int, int],
        w_value_range: tuple[int, int],
    ) -> None:
        macro = _make_macro(
            input_num=64,
            max_active_num=16,
            output_num=8,
            x_value_range=x_value_range,
            w_value_range=w_value_range,
        )
        macro.to(device)
        _, x = _random_operands(macro, batch=5, seed=202, device=device)
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        oracle = _plane_dot_oracle(macro, planes)
        y = macro.vec_mat_mul(planes.to(device), quantization_mode=0, adc_active_bits=0)
        assert y.device.type == device.type
        assert torch.equal(y.cpu(), oracle)


class TestFastPathQuantized:
    """Finite `adc_active_bits`: per-plane codes byte-identical to the int64 path."""

    def _quantized_macro(self) -> IdealCimMacro:
        # Per-plane dots span [-1920, 1920], inside the default window.
        return _make_macro(
            input_num=64,
            max_active_num=16,
            output_num=8,
            x_value_range=(-8, 7),
            w_value_range=(-15, 15),
            adc_bits=6,
        )

    def test_codes_match_forced_fallback_cpu(self) -> None:
        macro_fast = self._quantized_macro()
        macro_ref = self._quantized_macro()
        macro_ref._fp32_exact = False
        _, x = _random_operands(macro_fast, batch=5, seed=303, device=torch.device("cpu"))
        _random_operands(macro_ref, batch=5, seed=303, device=torch.device("cpu"))
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        quantization_mode, adc_active_bits = 0, 4
        y_fast = macro_fast.vec_mat_mul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        y_ref = macro_ref.vec_mat_mul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        assert y_fast.dtype == y_ref.dtype == torch.int32
        assert torch.equal(y_fast, y_ref)

    def test_codes_gpu_match_cpu_oracle(self, device: torch.device) -> None:
        macro = self._quantized_macro()
        w, x = _random_operands(macro, batch=5, seed=404, device=torch.device("cpu"))
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        quantization_mode, adc_active_bits = 0, 4
        y_cpu = macro.vec_mat_mul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        macro.to(device)
        macro.program(w.to(device))
        y_dev = macro.vec_mat_mul(
            planes.to(device), quantization_mode=quantization_mode, adc_active_bits=adc_active_bits
        )
        assert torch.equal(y_dev.cpu(), y_cpu)

    def test_training_jitter_rng_stream_identical_across_paths(self) -> None:
        """Identical fp32 inputs consume the RNG identically on both paths."""
        macro_fast = self._quantized_macro()
        macro_ref = self._quantized_macro()
        macro_ref._fp32_exact = False
        macro_fast.train()
        macro_ref.train()
        _, x = _random_operands(macro_fast, batch=5, seed=505, device=torch.device("cpu"))
        _random_operands(macro_ref, batch=5, seed=505, device=torch.device("cpu"))
        planes = _masked_planes(x, input_num=64, max_active_num=16)
        quantization_mode, adc_active_bits = 0, 4
        torch.manual_seed(7)
        y_fast = macro_fast.vec_mat_mul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        torch.manual_seed(7)
        y_ref = macro_ref.vec_mat_mul(planes, quantization_mode=quantization_mode, adc_active_bits=adc_active_bits)
        assert torch.equal(y_fast, y_ref)


class TestFallbackTrigger:
    """Bound at or above `2^24` keeps the int64 path (and stays exact)."""

    def test_flag_disabled_and_exact_beyond_fp32(self) -> None:
        # _max_plane_dot_abs = 3 * 2^23 * 1 >= 2^24 -> fallback.
        macro = _make_macro(
            input_num=3,
            max_active_num=3,
            output_num=2,
            x_value_range=(0, 1),
            w_value_range=(0, 2**23),
        )
        assert macro._fp32_exact is False
        # Dot = 2^24 + 1 is NOT fp32-representable: the fast path would
        # round it; the int64 path must carry it exactly.
        assert torch.tensor(2**24 + 1, dtype=torch.float32).item() == 2**24
        w = torch.tensor([[2**23, 0], [2**23, 0], [1, 0]], dtype=torch.int32)
        macro.program(w)
        x = torch.ones(3, dtype=torch.int32)
        y = macro.vec_mat_mul(x, quantization_mode=0, adc_active_bits=0)
        assert y.shape == (2,)
        assert y[0].item() == 2**24 + 1
        assert y[1].item() == 0

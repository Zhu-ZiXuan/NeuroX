"""Round-trip and shape-broadcast tests for ``CanonicalTranscoder``."""

import pytest
import torch

from neurox.mapper.transcoder import CanonicalTranscoder


def _canonical_representable_range(radix: int, digits: int) -> tuple[int, int]:
    max_abs = sum((radix - 1) * (radix**power) for power in range(digits - 1, -1, -2))
    return -max_abs, max_abs


class TestCanonicalEncoder:
    def test_known_case_radix4(self) -> None:
        min_val, max_val = _canonical_representable_range(radix=4, digits=4)
        x = torch.tensor([63], dtype=torch.int32)

        assert min_val <= int(x.item()) <= max_val, "input outside representable range"

        tc = CanonicalTranscoder(radix=4, digit_num=4)
        encoded = tc.encode(x)
        expected = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)

        assert torch.equal(encoded, expected), f"unexpected encoding: expected {expected}, got {encoded}"
        assert torch.equal(tc.decode(encoded), x), "decode does not match original input"

    def test_zero_negatives_and_boundaries(self) -> None:
        radix, digits = 4, 4
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.tensor([0, -1, -63, -15, min_val, max_val], dtype=torch.int32)

        assert torch.all((x >= min_val) & (x <= max_val)), "input outside representable range"

        tc = CanonicalTranscoder(radix=radix, digit_num=digits)
        encoded = tc.encode(x)
        decoded_x = tc.decode(encoded)
        assert torch.equal(decoded_x, x), "round-trip failed for zero / negative inputs"
        assert torch.all(encoded[0] == 0), "encoding of 0 must be all zeros"

    @pytest.mark.parametrize("radix", [2, 3, 4, 8])
    @pytest.mark.parametrize("digits", [4, 8])
    def test_full_range_fuzzing(self, radix: int, digits: int) -> None:
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.randint(min_val, max_val + 1, size=(1000,), dtype=torch.int32)

        tc = CanonicalTranscoder(radix=radix, digit_num=digits)
        encoded = tc.encode(x)
        decoded_x = tc.decode(encoded)

        assert torch.equal(decoded_x, x.to(torch.int64)), f"fuzzing failed at radix={radix}, digits={digits}"
        assert torch.all(encoded >= -(radix - 1)), "canonical digit underflowed below -(radix-1)"
        assert torch.all(encoded <= (radix - 1)), "canonical digit overflowed above (radix-1)"

    def test_tensor_broadcasting_and_shapes(self) -> None:
        shape = (16, 3, 3, 3)
        radix, digits = 4, 5
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.randint(min_val, max_val + 1, size=shape, dtype=torch.int32)

        tc = CanonicalTranscoder(radix=radix, digit_num=digits)
        encoded = tc.encode(x)

        expected_shape = (*shape, digits)
        assert encoded.shape == expected_shape, (
            f"unexpected encoded shape: expected {expected_shape}, got {encoded.shape}"
        )
        decoded_x = tc.decode(encoded)
        assert torch.equal(decoded_x, x.to(torch.int64)), "round-trip failed for multi-dim tensor"

    def test_torch_compile_compatibility(self) -> None:
        if not hasattr(torch, "compile"):
            pytest.skip("PyTorch version does not support torch.compile")

        radix, digits = 4, 6
        min_val, max_val = _canonical_representable_range(radix=radix, digits=digits)
        x = torch.randint(min_val, max_val + 1, size=(128, 128), dtype=torch.int32)

        tc = CanonicalTranscoder(radix=radix, digit_num=digits)

        def encode_fn(t: torch.Tensor) -> torch.Tensor:
            return tc.encode(t)

        compiled_encode = torch.compile(encode_fn, backend="inductor")
        _ = compiled_encode(x)
        expected = tc.encode(x)
        compiled = compiled_encode(x)

        assert torch.equal(compiled, expected), "torch.compile result diverges from eager"

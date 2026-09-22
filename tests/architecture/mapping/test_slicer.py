"""Value-domain slicing preserves linear results and enabled recovery costs."""

from __future__ import annotations

import pytest
import torch

from neurox import Profiler
from neurox.architecture.mapping.slicer import SimpleSlicer
from neurox.encoding import Encoding
from neurox.primitive.digital import (
    DigitalPolicy,
    RadixAccumulator,
    RadixAccumulatorConfig,
    RadixSummator,
    RadixSummatorConfig,
)


@pytest.mark.parametrize(
    ("encoding", "slice_num", "slice_value_range", "expected_radix"),
    [
        (Encoding.UNSIGNED, 3, (0, 7), 8),
        (Encoding.UNSIGNED, 2, (-3, 3), 4),
        (Encoding.CANONICAL, 2, (-4, 3), 4),
        (Encoding.TRUE_FORM, 2, (-2, 7), 3),
        (Encoding.TRUE_FORM, 2, (-7, 2), 3),
    ],
)
def test_simple_slicer_roundtrip_for_encoding_and_geometry_cases(
    encoding: Encoding,
    slice_num: int,
    slice_value_range: tuple[int, int],
    expected_radix: int,
    device: torch.device,
) -> None:
    slicer = SimpleSlicer(
        slice_num=slice_num,
        slice_value_range=slice_value_range,
        encoding=encoding,
    )
    assert slicer.slice_radix == expected_radix
    lo, hi = slicer.value_range
    values = torch.arange(lo, hi + 1, dtype=torch.int32, device=device)
    sliced = slicer.slice(values)
    assert sliced.shape == (hi - lo + 1, slice_num)
    assert sliced.min() >= slice_value_range[0]
    assert sliced.max() <= slice_value_range[1]
    decoded = slicer.recover(sliced, dim=-1)
    assert torch.equal(decoded, values)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("encoding", [Encoding.TRUE_FORM, Encoding.CANONICAL])
def test_recover_recombines_linear_results(dim: int, encoding: Encoding, device: torch.device) -> None:
    slicer = SimpleSlicer(slice_num=4, slice_value_range=(-3, 3), encoding=encoding)
    values = torch.tensor([[-5, 2, 7], [1, -3, 4]], dtype=torch.int32, device=device)
    # Results may exceed the range of a single slice after a linear operation.
    input_dim = -2 if dim == -1 else -1
    partial = slicer.slice(values, dim=dim).sum(dim=input_dim, keepdim=True, dtype=values.dtype)
    actual = slicer.recover(partial, dim=dim)
    expected = values.sum(dim=-1, keepdim=True, dtype=values.dtype)
    torch.testing.assert_close(actual, expected)

    enable = torch.tensor([True, False], device=device).view(2, 1, 1).movedim(1, dim)
    masked = slicer.recover(partial, dim=dim, enable=enable)
    expected[1] = 0
    torch.testing.assert_close(masked, expected)


@pytest.mark.parametrize("serial", [False, True])
def test_signed_slice_results_reach_circuit_wrap_and_enabled_operation_accounting(
    serial: bool, device: torch.device
) -> None:
    circuit_type = RadixAccumulator if serial else RadixSummator
    config_type = RadixAccumulatorConfig if serial else RadixSummatorConfig
    circuit = circuit_type(
        config=config_type(bit_width=4, energy_per_op__fJ=2.0, area_per_inst__um2=0.0, leakage_per_inst__uW=0.0),
        policy=DigitalPolicy(),
        inst_shape=(),
    ).to(device)
    slicer = SimpleSlicer(slice_num=3, slice_value_range=(-1, 1), encoding=Encoding.TRUE_FORM, recovery_circuit=circuit)
    values = torch.tensor([[3, 1, -5], [0, 0, 0], [1, 1, -1]], dtype=torch.int64, device=device)
    enable = torch.tensor([[True], [True], [False]], device=device)
    circuit.set_profile_leading_rank(1)
    profiler = Profiler(concat_dim=0, sync_device=device)
    profiler.collect_static_data(circuit)
    with profiler:
        actual = slicer.recover(values, dim=-1, enable=enable)
    # The first complete signed sum is -15, wrapping to 1.
    torch.testing.assert_close(actual, torch.tensor([1, 0, 0], device=device))
    expected_energy = torch.tensor([6.0, 6.0, 0.0], device=device)
    torch.testing.assert_close(profiler.result[""].dynamic_energy__fJ, expected_energy, check_dtype=False)

"""Persisted observations from ADC-input characterization."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from neurox.common.dataclass_mixin import TensorDataClassMixin
from neurox.tools.output import atomic_output


class AdcProbeData(TensorDataClassMixin):
    input_name: str
    input_value: Tensor
    """ADC decision input, CPU float32.
    Shape: `[sample]`."""
    ideal_value: Tensor
    """Lossless macro result, CPU int64.
    Shape: `[sample]`."""
    ideal_value_support: tuple[int, ...]
    """Reachable exact results under the declared value domains."""

    @property
    def ideal_value_range(self) -> tuple[int, int]:
        return self.ideal_value_support[0], self.ideal_value_support[-1]


def save_adc_probe_data(data: AdcProbeData, path: Path) -> None:
    """Atomically replace an artifact with paired ADC observations in format 2.

    Parent directories are created as needed. Tensor fields are serialized as
    supplied; the function does not normalize dtype or move them to CPU. Provide
    the aligned observation layout described by `AdcProbeData`. A failed write
    leaves an existing artifact intact. Use `load_adc_probe_data` to read it.

    Args:
        data: Aligned probe observations to serialize without normalization.
        path: Destination artifact path; parent directories are created as
            needed.
    """
    with atomic_output(path) as temporary:
        torch.save(
            {
                "format_version": 2,
                "input_name": data.input_name,
                "input_value": data.input_value,
                "ideal_value": data.ideal_value,
                "ideal_value_support": data.ideal_value_support,
            },
            temporary,
        )


def load_adc_probe_data(path: Path) -> AdcProbeData:
    """Load paired observations onto CPU from a versioned probe artifact.

    Use PyTorch's weights-only loader. Formats 1 and 2 are accepted; format 1's
    inclusive ideal-value range is expanded into its integer support. Tensor
    layouts and dtypes are retained, without recomputing observations or fitting
    statistics. Unsupported format versions raise `ValueError`; malformed
    payload and file errors propagate.

    Args:
        path: Path to a version 1 or 2 ADC probe artifact.

    Returns:
        Paired observations on CPU with the saved layouts and dtypes.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") not in (1, 2):
        raise ValueError(f"unsupported ADC probe data format in {path}")
    if payload["format_version"] == 1:
        lower, upper = payload["ideal_value_range"]
        ideal_value_support = tuple(range(lower, upper + 1))
    else:
        ideal_value_support = tuple(payload["ideal_value_support"])
    return AdcProbeData(
        input_name=payload["input_name"],
        input_value=payload["input_value"],
        ideal_value=payload["ideal_value"],
        ideal_value_support=ideal_value_support,
    )

"""Stored observations from ADC-input characterization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import Tensor

from neurox.common import TensorDataClassBase


class AdcProbeData(TensorDataClassBase):
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
    """Persist paired observations for reusable offline analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
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
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_adc_probe_data(path: Path) -> AdcProbeData:
    """Load observations written by `save_adc_probe_data`."""
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

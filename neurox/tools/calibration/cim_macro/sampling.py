"""Synthetic logical-workload sampling for calibration tools.

Scheme-agnostic: value ranges come from the `CimMacro` base surface, while
logical dimensions are explicit tool inputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.common.serialize import dict_from_file
from neurox.common.tensor_dataclass_mixin import TensorDataClassMixin
from neurox.primitive.macro.cim import CimMacro, CimMacroConfig, CimMacroPolicy

__all__ = [
    "AxisDistribution",
    "Distribution",
    "load_distribution",
    "make_generator",
    "sample_w",
    "sample_x_batches",
]


class AxisDistribution(TensorDataClassMixin):
    """One axis's sampling distribution.

    Values and probabilities are one object because neither is meaningful
    alone: a sample is drawn by indexing `values` with a `probs` draw.
    """

    values: Tensor
    """Allowed values, int64.
    Shape: `[value]`."""
    probs: Tensor
    """Normalized probabilities matching `values`, float64.
    Shape: `[value]`."""


class Distribution(TensorDataClassMixin):
    """Parsed synthetic-workload distribution.

    A `None`-valued axis means uniform sampling over the macro's legal integer
    range on that axis.
    """

    w: AxisDistribution | None
    """Distribution over the allowed per-cell digit values."""
    x: AxisDistribution | None
    """Distribution over the allowed per-row input codes."""
    source: str
    """Human-readable provenance — `uniform` or the TOML path."""


def load_distribution(path: Path | None, macro: CimMacro[CimMacroConfig, CimMacroPolicy]) -> Distribution:
    """Load a synthetic-workload distribution TOML.

    A `None` path means fully uniform; a present file may omit `[w]` or `[x]`
    to keep that axis uniform. The schema is

        [w]
        values = [-1, 0, 1]
        probs = [0.20, 0.65, 0.15]

        [x]
        values = [0, 1, 2, 3]
        probs = [0.70, 0.20, 0.08, 0.02]

    Args:
        path: TOML path, or `None` for uniform sampling.
        macro: Built macro supplying the legal value ranges the value sets are
            validated against.

    Returns:
        Validated distribution with internally normalized probabilities.

    Raises:
        TypeError: A section is not a table, or a `values` / `probs` entry is
            not of the required scalar type.
        ValueError: An unknown top-level key, or a malformed section — a
            missing or empty list, length mismatch, negative or all-zero
            probability, a value outside the macro's legal range.
    """
    if path is None:
        return Distribution(w=None, x=None, source="uniform")

    raw = dict_from_file(path)
    unknown = sorted(k for k in raw if k not in ("w", "x"))
    if unknown:
        raise ValueError(
            f"distribution TOML at {path}: unknown top-level key(s) {unknown}; only '[w]' and '[x]' are recognised"
        )
    return Distribution(
        w=_load_axis(raw, "w", macro.w_value_range),
        x=_load_axis(raw, "x", macro.x_value_range),
        source=str(path),
    )


def _load_axis(
    raw: dict[str, Any],
    key: str,
    legal_range: tuple[int, int],
) -> AxisDistribution | None:
    """Parse one axis section into an `AxisDistribution`.

    An absent section yields `None`.
    """
    if key not in raw:
        return None

    section = raw[key]
    if not isinstance(section, dict):
        raise TypeError(f"distribution [{key}]: expected a TOML table, got {type(section).__name__}")
    if "values" not in section or "probs" not in section:
        raise ValueError(f"distribution [{key}]: both 'values' and 'probs' are required")

    values_raw = section["values"]
    probs_raw = section["probs"]

    if not isinstance(values_raw, list):
        raise TypeError(f"distribution [{key}].values: must be a list, got {type(values_raw).__name__}")
    for i, v in enumerate(values_raw):
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"distribution [{key}].values: entry {i} must be an integer; got {v!r}")
    if not isinstance(probs_raw, list):
        raise TypeError(f"distribution [{key}].probs: must be a list, got {type(probs_raw).__name__}")
    for i, p in enumerate(probs_raw):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(f"distribution [{key}].probs: entry {i} must be a number; got {p!r}")

    if len(values_raw) == 0:
        raise ValueError(f"distribution [{key}].values: must be non-empty")
    if len(values_raw) != len(probs_raw):
        raise ValueError(f"distribution [{key}]: len(values)={len(values_raw)} != len(probs)={len(probs_raw)}")

    probs = [float(p) for p in probs_raw]
    if any(p < 0.0 for p in probs):
        raise ValueError(f"distribution [{key}].probs: every entry must be >= 0")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError(f"distribution [{key}].probs: at least one entry must be > 0")

    lo, hi = legal_range
    for v in values_raw:
        if not (lo <= v <= hi):
            raise ValueError(f"distribution [{key}].values: value {v} outside macro legal range [{lo}, {hi}]")

    return AxisDistribution(
        values=torch.tensor(values_raw, dtype=torch.int64),
        probs=torch.tensor([p / total for p in probs], dtype=torch.float64),
    )


def sample_w(
    distribution: Distribution,
    macro: CimMacro[CimMacroConfig, CimMacroPolicy],
    *,
    input_num: int,
    output_num: int,
    n: int,
    batch_w: int = 1,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield `n // batch_w` batches of logical weight matrices.

    `n` must be a multiple of `batch_w`: every yielded tensor carries a fixed
    leading `batch_w` axis to match the macro's `inst_shape=(batch_w,)`
    contract, so a partial final batch fails the `program(w)` shape check. A
    call site wanting "at least N" coverage rounds `n` up to the next multiple
    of `batch_w`.

    Yields:
        Weight tensor.
        Shape: `[weight_sample, input, output]`.

    Raises:
        ValueError: A non-positive `batch_w`, or an `n` that is not a multiple
            of it.
    """
    if batch_w <= 0:
        raise ValueError(f"batch_w ({batch_w}) must be > 0")
    if n % batch_w != 0:
        raise ValueError(
            f"sample_w: n ({n}) must be a multiple of batch_w ({batch_w}); "
            "partial final batches would break the macro's fixed inst_shape contract"
        )
    shape_per = (batch_w, input_num, output_num)
    n_per = math.prod(shape_per)
    num_yields = n // batch_w
    for _ in range(num_yields):
        if distribution.w is None:
            lo, hi = macro.w_value_range
            yield torch.randint(
                lo,
                hi + 1,
                shape_per,
                device=device,
                dtype=torch.int64,
                generator=generator,
            )
        else:
            probs = distribution.w.probs.to(device)
            values = distribution.w.values.to(device)
            idx = torch.multinomial(probs, num_samples=n_per, replacement=True, generator=generator)
            yield values[idx].view(shape_per).to(torch.int64)


def sample_x_batches(
    distribution: Distribution,
    macro: CimMacro[CimMacroConfig, CimMacroPolicy],
    *,
    input_num: int,
    n_total: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield input batches summing to `n_total` vectors.

    Yields:
        Int64 batch on `device`; the leading axis is `batch_size` except on the
        last batch, which carries whatever remains of `n_total`.
        Shape: `[sample, input]`.

    Raises:
        ValueError: A non-positive `batch_size` or a negative `n_total`.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > 0")
    if n_total < 0:
        raise ValueError(f"n_total ({n_total}) must be >= 0")
    remaining = n_total
    while remaining > 0:
        cur = min(batch_size, remaining)
        if distribution.x is None:
            lo, hi = macro.x_value_range
            x = torch.randint(
                lo,
                hi + 1,
                (cur, input_num),
                device=device,
                dtype=torch.int64,
                generator=generator,
            )
        else:
            probs = distribution.x.probs.to(device)
            values = distribution.x.values.to(device)
            idx = torch.multinomial(
                probs,
                num_samples=cur * input_num,
                replacement=True,
                generator=generator,
            )
            x = values[idx].view(cur, input_num).to(torch.int64)
        yield x
        remaining -= cur


def make_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
    """Build a deterministic generator on `device`, or `None` for an unset seed."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

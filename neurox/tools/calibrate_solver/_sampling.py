"""Synthetic-workload sampling backend for the solver calibration tools.

Scheme-agnostic: value ranges come from the :class:`CimMacro` base surface,
while logical dimensions are explicit tool inputs.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurox.common.serialize import dict_from_file
from neurox.primitive.macro.cim import CimMacro


@dataclass(frozen=True)
class Distribution:
    """Parsed synthetic-workload distribution.

    ``None``-valued fields mean uniform sampling over the xbar's
    legal integer range on that axis.

    Attributes:
        w_values: Allowed per-cell digit values, int64 1-D, or ``None``.
        w_probs: Normalized probabilities matching ``w_values``,
            float64 1-D, or ``None``.
        x_values: Allowed per-row input codes, int64 1-D, or ``None``.
        x_probs: Normalized probabilities matching ``x_values``,
            float64 1-D, or ``None``.
        source: Human-readable provenance — ``"uniform"`` or the TOML path.
    """

    w_values: Tensor | None
    w_probs: Tensor | None
    x_values: Tensor | None
    x_probs: Tensor | None
    source: str


def load_distribution(path: Path | None, xbar: CimMacro) -> Distribution:
    """Load a synthetic-workload distribution TOML.

    ``None`` means fully uniform; a present file may omit ``[w]`` or
    ``[x]`` to keep that axis uniform.

    Schema::

        [w]
        values = [-1, 0, 1]
        probs = [0.20, 0.65, 0.15]

        [x]
        values = [0, 1, 2, 3]
        probs = [0.70, 0.20, 0.08, 0.02]

    Args:
        path: TOML path or ``None``.
        xbar: Built macro supplying ``w_value_range`` / ``x_value_range`` for
            value-set validation.

    Returns:
        Validated, internally-normalised :class:`Distribution`.

    Raises:
        ValueError: When a section is malformed (length mismatch,
            negative probability, all-zero probability, value outside
            the xbar's legal range, etc.).
    """
    if path is None:
        return Distribution(
            w_values=None,
            w_probs=None,
            x_values=None,
            x_probs=None,
            source="uniform",
        )

    raw = dict_from_file(path)
    unknown = sorted(k for k in raw if k not in ("w", "x"))
    if unknown:
        raise ValueError(
            f"distribution TOML at {path}: unknown top-level key(s) {unknown}; only '[w]' and '[x]' are recognised"
        )
    w_values, w_probs = _load_axis(raw, "w", xbar.w_value_range)
    x_values, x_probs = _load_axis(raw, "x", xbar.x_value_range)
    return Distribution(
        w_values=w_values,
        w_probs=w_probs,
        x_values=x_values,
        x_probs=x_probs,
        source=str(path),
    )


def _load_axis(
    raw: dict[str, Any],
    key: str,
    legal_range: tuple[int, int],
) -> tuple[Tensor | None, Tensor | None]:
    """Return ``(values_int64, normalized_probs_float64)`` or ``(None, None)``."""
    if key not in raw:
        return None, None

    section = raw[key]
    if not isinstance(section, dict):
        raise ValueError(f"distribution [{key}]: expected a TOML table, got {type(section).__name__}")
    if "values" not in section or "probs" not in section:
        raise ValueError(f"distribution [{key}]: both 'values' and 'probs' are required")

    values_raw = section["values"]
    probs_raw = section["probs"]

    if not isinstance(values_raw, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in values_raw):
        raise ValueError(f"distribution [{key}].values: must be a list of integers")
    if not isinstance(probs_raw, list) or not all(
        isinstance(p, (int, float)) and not isinstance(p, bool) for p in probs_raw
    ):
        raise ValueError(f"distribution [{key}].probs: must be a list of numbers")

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
            raise ValueError(f"distribution [{key}].values: value {v} outside xbar legal range [{lo}, {hi}]")

    values = torch.tensor(values_raw, dtype=torch.int64)
    probs_tensor = torch.tensor([p / total for p in probs], dtype=torch.float64)
    return values, probs_tensor


def sample_w(
    distribution: Distribution,
    macro: CimMacro,
    *,
    input_num: int,
    output_num: int,
    n: int,
    batch_w: int = 1,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield ``n // batch_w`` batches of logical weight matrices.

    ``n`` **must** be a multiple of ``batch_w`` — every yielded tensor
    carries a fixed leading ``batch_w`` axis to match the xbar's
    ``inst_shape=(batch_w,)`` contract, so a partial final batch would
    immediately fail ``xbar.program(w)``'s shape check. Round ``n`` up
    to the next multiple of ``batch_w`` at the call site if you need
    "at least N" coverage.

    Yields:
        The leading axis is present only when ``batch_w > 1``; with
        ``batch_w == 1`` (default) each yielded tensor drops it.
        Shape: ``[batch_w, input_num, output_num]``.
    """
    if batch_w <= 0:
        raise ValueError(f"batch_w ({batch_w}) must be > 0")
    if n % batch_w != 0:
        raise ValueError(
            f"sample_w: n ({n}) must be a multiple of batch_w ({batch_w}); "
            "partial final batches would break the xbar's fixed inst_shape contract"
        )
    leading = () if batch_w == 1 else (batch_w,)
    shape_per = (*leading, input_num, output_num)
    n_per = math.prod(shape_per)
    num_yields = n // batch_w
    for _ in range(num_yields):
        if distribution.w_values is None:
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
            assert distribution.w_probs is not None  # paired with w_values via _load_axis
            probs = distribution.w_probs.to(device)
            values = distribution.w_values.to(device)
            idx = torch.multinomial(probs, num_samples=n_per, replacement=True, generator=generator)
            yield values[idx].view(shape_per).to(torch.int64)


def sample_x_batches(
    distribution: Distribution,
    macro: CimMacro,
    *,
    input_num: int,
    n_total: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Iterator[Tensor]:
    """Yield input batches summing to ``n_total`` vectors.

    Yields:
        Each batch is int64 on ``device``; its leading axis is ``batch_size``
        except on the last batch, which carries whatever remains of ``n_total``.
        Shape: ``[batch_size, input_num]``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be > 0")
    if n_total < 0:
        raise ValueError(f"n_total ({n_total}) must be >= 0")
    remaining = n_total
    while remaining > 0:
        cur = min(batch_size, remaining)
        if distribution.x_values is None:
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
            assert distribution.x_probs is not None  # paired with x_values via _load_axis
            probs = distribution.x_probs.to(device)
            values = distribution.x_values.to(device)
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
    """Build a deterministic generator on ``device`` when ``seed`` is set."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

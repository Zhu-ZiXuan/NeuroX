"""File-format authority for the two calibration-exchange TOML formats.

Emit, parse, and validation for both formats live only here:

- **Layer-range mapping** (:func:`load_layer_ranges`) — the
  :mod:`.mode_derive` input. Top-level quoted layer-name keys::

      "layer.name" = { range = 12.5, signed = true }

  ``range`` is the layer's ADC design range (finite, > 0); ``signed``
  whether the layer quantizes a signed input range. Producing this file
  (e.g. extracting learned range params from a training checkpoint) is a
  consumer-side step outside the tools.

- **Mode set** (:func:`load_mode_set` / :func:`dump_mode_set`) — the
  :mod:`.mode_derive` output and the single mode source for
  :mod:`.threshold_probe` and :mod:`.rescale_fit`::

      [[modes]]
      adc_mode = 0
      signed = false
      range = 12.5
      layer_num = 3

      [layers]
      "layer.name" = 0

  ``[[modes]]`` tables ascend by ``adc_mode`` (contiguous from 0);
  ``[layers]`` maps every covered layer to its mode.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _check_range(value: object, *, where: str) -> float:
    """Validate one ``range`` value: a finite float > 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: range must be a number; got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{where}: require range finite and > 0; got {v!r}")
    return v


def _check_bool(value: object, *, where: str, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: {key} must be a bool; got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Layer-range mapping (mode_derive input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRange:
    """One layer's ADC design range and signedness.

    Attributes:
        range: Design range magnitude (finite, > 0).
        signed: Whether the layer quantizes a signed input range.
    """

    range: float
    signed: bool

    def __post_init__(self) -> None:
        _check_range(self.range, where="LayerRange")
        _check_bool(self.signed, where="LayerRange", key="signed")


def load_layer_ranges(path: Path) -> dict[str, LayerRange]:
    """Parse and validate a layer-range mapping TOML.

    Returns:
        ``layer name -> LayerRange`` for every entry, in file order.

    Raises:
        ValueError: On an empty mapping, a non-table entry, an unknown or
            missing key, or an invalid ``range`` / ``signed`` value.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if not raw:
        raise ValueError(f"{path}: require at least one layer entry")
    out: dict[str, LayerRange] = {}
    for name, entry in raw.items():
        where = f"{path}: layer {name!r}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: entry must be a table {{ range = ..., signed = ... }}; got {entry!r}")
        if set(entry) != {"range", "signed"}:
            raise ValueError(f"{where}: entry keys must be exactly {{range, signed}}; got {sorted(entry)}")
        out[name] = LayerRange(
            range=_check_range(entry["range"], where=where),
            signed=_check_bool(entry["signed"], where=where, key="signed"),
        )
    return out


# ---------------------------------------------------------------------------
# Mode set (mode_derive output; threshold_probe / rescale_fit input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdcMode:
    """One ADC operating mode of the mode set.

    Attributes:
        adc_mode: Operating-mode index (enumeration position, from 0).
        signed: Whether the mode serves signed layers.
        range: Mode design range — covers every member layer's range.
        layer_num: Number of layers assigned to the mode (>= 1).
    """

    adc_mode: int
    signed: bool
    range: float
    layer_num: int

    def __post_init__(self) -> None:
        if self.adc_mode < 0:
            raise ValueError(f"require: adc_mode ({self.adc_mode}) >= 0")
        _check_bool(self.signed, where=f"mode {self.adc_mode}", key="signed")
        _check_range(self.range, where=f"mode {self.adc_mode}")
        if self.layer_num < 1:
            raise ValueError(f"require: mode {self.adc_mode} layer_num ({self.layer_num}) >= 1")


@dataclass(frozen=True)
class ModeSet:
    """Validated mode set: the mode tables plus the layer -> mode mapping.

    Attributes:
        modes: Modes ascending by ``adc_mode``, contiguous from 0.
        layers: ``layer name -> adc_mode`` for every covered layer; per
            mode the assignment count equals its ``layer_num``.
    """

    modes: tuple[AdcMode, ...]
    layers: dict[str, int]

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("require: at least one mode")
        if [m.adc_mode for m in self.modes] != list(range(len(self.modes))):
            raise ValueError(f"require: adc_mode contiguous from 0; got {[m.adc_mode for m in self.modes]}")
        counts = {m.adc_mode: 0 for m in self.modes}
        for name, mode in self.layers.items():
            if mode not in counts:
                raise ValueError(f"layer {name!r} maps to unknown adc_mode {mode}")
            counts[mode] += 1
        for m in self.modes:
            if counts[m.adc_mode] != m.layer_num:
                raise ValueError(
                    f"mode {m.adc_mode}: layer_num ({m.layer_num}) != assigned layer count ({counts[m.adc_mode]})"
                )


def load_mode_set(path: Path) -> ModeSet:
    """Parse and validate a mode-set TOML.

    Raises:
        ValueError: On a missing / malformed ``[[modes]]`` list or
            ``[layers]`` table, or any :class:`ModeSet` invariant violation.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if set(raw) != {"modes", "layers"}:
        raise ValueError(f"{path}: top-level keys must be exactly {{modes, layers}}; got {sorted(raw)}")
    if not isinstance(raw["modes"], list):
        raise ValueError(f"{path}: [[modes]] must be an array of tables")
    modes: list[AdcMode] = []
    for i, entry in enumerate(raw["modes"]):
        where = f"{path}: [[modes]] entry {i}"
        if not isinstance(entry, dict) or set(entry) != {"adc_mode", "signed", "range", "layer_num"}:
            raise ValueError(f"{where}: keys must be exactly {{adc_mode, signed, range, layer_num}}; got {entry!r}")
        if not isinstance(entry["adc_mode"], int) or isinstance(entry["adc_mode"], bool):
            raise ValueError(f"{where}: adc_mode must be an int; got {entry['adc_mode']!r}")
        if not isinstance(entry["layer_num"], int) or isinstance(entry["layer_num"], bool):
            raise ValueError(f"{where}: layer_num must be an int; got {entry['layer_num']!r}")
        modes.append(
            AdcMode(
                adc_mode=entry["adc_mode"],
                signed=_check_bool(entry["signed"], where=where, key="signed"),
                range=_check_range(entry["range"], where=where),
                layer_num=entry["layer_num"],
            )
        )
    layers_raw = raw["layers"]
    if not isinstance(layers_raw, dict):
        raise ValueError(f"{path}: [layers] must be a table of 'layer name' -> adc_mode")
    layers: dict[str, int] = {}
    for name, mode in layers_raw.items():
        if not isinstance(mode, int) or isinstance(mode, bool):
            raise ValueError(f"{path}: [layers] {name!r} must map to an int adc_mode; got {mode!r}")
        layers[name] = mode
    return ModeSet(modes=tuple(modes), layers=layers)


def dump_mode_set(mode_set: ModeSet) -> str:
    """Serialize a validated mode set to its TOML text (ends with a newline)."""
    lines: list[str] = []
    for m in mode_set.modes:
        lines += [
            "[[modes]]",
            f"adc_mode = {m.adc_mode}",
            f"signed = {'true' if m.signed else 'false'}",
            f"range = {float(m.range)!r}",
            f"layer_num = {m.layer_num}",
            "",
        ]
    lines.append("[layers]")
    # json.dumps escaping is valid TOML basic-string escaping.
    lines += [f"{json.dumps(name)} = {mode}" for name, mode in mode_set.layers.items()]
    return "\n".join(lines) + "\n"

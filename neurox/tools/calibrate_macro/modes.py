"""Parse, validate, and emit macro-mode exchange TOML formats.

Emit, parse, and validation for both formats live only here.

The layer-range mapping is keyed by quoted top-level layer names:

    "layer.name" = { range = [-12.5, 12.5] }

`range` is the layer's inclusive design range `[lo, hi]` in MAC units
(finite floats, `lo <= hi`, `hi > 0`); `lo < 0` marks a layer whose
quantization input is signed.

The mode set groups those ranges under canonical quantization windows:

    [[modes]]
    quantization_mode = 0
    quantization_input_range = [-16, 15]
    layer_num = 3

    [layers]
    "layer.name" = 0

The `[[modes]]` tables ascend by `quantization_mode`, contiguous from 0, and
carry the canonical inclusive window the mode quantizes; `[layers]` maps every
covered layer to its mode.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from neurox.common import ValidateMixin
from neurox.primitive.macro.cim import validate_quantization_input_range


def _check_float(value: object, *, where: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where}: {key} must be a number; got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{where}: require {key} finite; got {v!r}")
    return v


def _check_int(value: object, *, where: str, key: str) -> int:
    # A TOML boolean is a Python int; it is not an integer entry here.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where}: {key} must be an int; got {value!r}")
    return value


def _check_pair(value: object, *, where: str, key: str) -> tuple[object, object]:
    if not isinstance(value, list):
        raise TypeError(f"{where}: {key} must be an array [lower, upper]; got {value!r}")
    if len(value) != 2:
        raise ValueError(f"{where}: {key} must have exactly 2 elements; got {len(value)}")
    return value[0], value[1]


@dataclass(frozen=True)
class LayerRange(ValidateMixin):
    """One layer's inclusive quantization design range."""

    range: tuple[float, float]
    """Inclusive design range `(lo, hi)` in MAC units; finite, `lo <= hi`,
    `hi > 0`."""

    def __post_init__(self) -> None:
        lo, hi = self.range
        self._require_le(lo, "LayerRange range lower", hi)
        self._require_pos(hi, "LayerRange range upper")

    @property
    def signed(self) -> bool:
        """Whether the range reaches below zero."""
        return self.range[0] < 0.0


def canonical_window(layer_range: LayerRange) -> tuple[int, int]:
    """Return the canonical integer window covering one layer's range.

    A non-negative range maps to the unsigned window `[0, ceil(hi)]`; a signed
    range maps to the mid-zero window `[-m, m - 1]` with
    `m = max(ceil(-lo), ceil(hi) + 1)`, the smallest mid-zero window whose
    inclusive bounds cover both sides.

    Raises:
        ValueError: The derived window is not canonical (an unreachable
            state kept as an assertion of the derivation).
    """
    lo, hi = layer_range.range
    if not layer_range.signed:
        window = (0, math.ceil(hi))
    else:
        m = max(math.ceil(-lo), math.ceil(hi) + 1)
        window = (-m, m - 1)
    validate_quantization_input_range(window)
    return window


def load_layer_ranges(path: Path) -> dict[str, LayerRange]:
    """Parse and validate a layer-range mapping TOML.

    Returns:
        `layer name -> LayerRange` for every entry, in file order.

    Raises:
        TypeError: An entry is not a table, or its `range` is not an array of
            numbers.
        ValueError: On an empty mapping, an unknown or missing key, or an
            invalid `range` value.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if not raw:
        raise ValueError(f"{path}: require at least one layer entry")
    out: dict[str, LayerRange] = {}
    for name, entry in raw.items():
        where = f"{path}: layer {name!r}"
        if not isinstance(entry, dict):
            raise TypeError(f"{where}: entry must be a table {{ range = [lo, hi] }}; got {entry!r}")
        if set(entry) != {"range"}:
            raise ValueError(f"{where}: entry keys must be exactly {{range}}; got {sorted(entry)}")
        lo, hi = _check_pair(entry["range"], where=where, key="range")
        out[name] = LayerRange(
            range=(
                _check_float(lo, where=where, key="range lower"),
                _check_float(hi, where=where, key="range upper"),
            )
        )
    return out


@dataclass(frozen=True)
class MacroMode(ValidateMixin):
    """One quantization operating mode of the mode set."""

    quantization_mode: int
    """Operating-mode index — the enumeration position, from 0."""
    quantization_input_range: tuple[int, int]
    """Canonical inclusive MAC-unit window `(lower, upper)` the mode
    quantizes; it covers every member layer's design range."""
    layer_num: int
    """Layers assigned to the mode, at least one."""

    def __post_init__(self) -> None:
        self._require_non_neg(self.quantization_mode, "quantization_mode")
        validate_quantization_input_range(self.quantization_input_range)
        self._require_pos(self.layer_num, f"mode {self.quantization_mode} layer_num")


@dataclass(frozen=True)
class ModeSet(ValidateMixin):
    """Validated mode set: the mode tables plus the layer -> mode mapping."""

    modes: tuple[MacroMode, ...]
    """Modes ascending by `quantization_mode`, contiguous from 0."""
    layers: dict[str, int]
    """`layer name -> quantization_mode` for every covered layer; per mode the
    assignment count equals its `layer_num`."""

    def __post_init__(self) -> None:
        self._require_non_empty(self.modes, "modes")
        if [m.quantization_mode for m in self.modes] != list(range(len(self.modes))):
            raise ValueError(
                f"require: quantization_mode contiguous from 0; got {[m.quantization_mode for m in self.modes]}"
            )
        counts = {m.quantization_mode: 0 for m in self.modes}
        for name, mode in self.layers.items():
            if mode not in counts:
                raise ValueError(f"layer {name!r} maps to unknown quantization_mode {mode}")
            counts[mode] += 1
        for m in self.modes:
            if counts[m.quantization_mode] != m.layer_num:
                raise ValueError(
                    f"mode {m.quantization_mode}: layer_num ({m.layer_num}) != "
                    f"assigned layer count ({counts[m.quantization_mode]})"
                )


def load_mode_set(path: Path) -> ModeSet:
    """Parse and validate a mode-set TOML.

    Raises:
        TypeError: `[[modes]]` is not an array of tables, a `[[modes]]` entry
            or `[layers]` is not a table, or a field carries the wrong type.
        ValueError: On a missing key, an unexpected key set, or any mode-set
            invariant violation.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if set(raw) != {"modes", "layers"}:
        raise ValueError(f"{path}: top-level keys must be exactly {{modes, layers}}; got {sorted(raw)}")
    if not isinstance(raw["modes"], list):
        raise TypeError(f"{path}: [[modes]] must be an array of tables")
    modes: list[MacroMode] = []
    for i, entry in enumerate(raw["modes"]):
        where = f"{path}: [[modes]] entry {i}"
        expected = {"quantization_mode", "quantization_input_range", "layer_num"}
        if not isinstance(entry, dict):
            raise TypeError(f"{where}: entry must be a table; got {entry!r}")
        if set(entry) != expected:
            raise ValueError(f"{where}: keys must be exactly {sorted(expected)}; got {entry!r}")
        lower, upper = _check_pair(entry["quantization_input_range"], where=where, key="quantization_input_range")
        modes.append(
            MacroMode(
                quantization_mode=_check_int(entry["quantization_mode"], where=where, key="quantization_mode"),
                quantization_input_range=(
                    _check_int(lower, where=where, key="quantization_input_range lower"),
                    _check_int(upper, where=where, key="quantization_input_range upper"),
                ),
                layer_num=_check_int(entry["layer_num"], where=where, key="layer_num"),
            )
        )
    layers_raw = raw["layers"]
    if not isinstance(layers_raw, dict):
        raise TypeError(f"{path}: [layers] must be a table of 'layer name' -> quantization_mode")
    layers: dict[str, int] = {}
    for name, mode in layers_raw.items():
        layers[name] = _check_int(mode, where=f"{path}: [layers] {name!r}", key="quantization_mode")
    return ModeSet(modes=tuple(modes), layers=layers)


def dump_mode_set(mode_set: ModeSet) -> str:
    """Serialize a validated mode set to its TOML text (ends with a newline)."""
    lines: list[str] = []
    for m in mode_set.modes:
        lower, upper = m.quantization_input_range
        lines += [
            "[[modes]]",
            f"quantization_mode = {m.quantization_mode}",
            f"quantization_input_range = [{lower}, {upper}]",
            f"layer_num = {m.layer_num}",
            "",
        ]
    lines.append("[layers]")
    # json.dumps escaping is valid TOML basic-string escaping.
    lines += [f"{json.dumps(name)} = {mode}" for name, mode in mode_set.layers.items()]
    return "\n".join(lines) + "\n"

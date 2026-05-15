"""Shared macro factories for NeuroX examples.

Every example CLI takes two hardware-related arguments:

- ``--config <path>`` — the chip TOML.  Defaults to
  :data:`neurox.config.DEFAULT_1T1R_TOML` (the bundled reference chip)
  but user-supplied files are supported.
- ``--xbar {physical,ideal}`` — the tile implementation.  ``physical``
  runs the full 1T1R circuit solver; ``ideal`` swaps in a lossless
  :class:`IdealXbar` twin derived from the physical tile via
  :meth:`Xbar.to_ideal`.

This module exposes two helpers used by every CLI:

- :func:`derive_quant_spec` — operator quantisation grid derived from
  the chosen TOML's ``(w_states, x_states, transcoder.digit_num)``.
- :func:`build_macro_factory` — returns a zero-argument callable that
  ``neurox.replace_for_hat`` / ``neurox.build_evaluator`` consume.

``IdealMacro`` is deliberately NOT exposed here.  It's a
test-and-validation helper kept in :mod:`neurox.macro.ideal` for unit
tests; production pipelines should always go through :class:`XbarMacro`.
"""

from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import torch

from neurox.analog import (
    AnalogMux,
    AnalogMuxConfig,
    Decoder,
    DecoderConfig,
    Driver,
    DriverConfig,
    GeneralDAC,
    GeneralDACConfig,
    OpAmpTIA,
    OpAmpTIAConfig,
    SwitchCap,
    SwitchCapConfig,
)
from neurox.analog.adc import McsSarAdc, McsSarAdcConfig
from neurox.analog.readout import OffsetSwitchCapMuxAdcReadOut, ReadOutConfig
from neurox.common import T_ROOM__K, dict_configs_from_file, dict_from_file
from neurox.config import DEFAULT_1T1R_TOML
from neurox.device import NMOS, RRAM, NMOSConfig, RRAMConfig, Wire, WireConfig
from neurox.digital import (
    Accumulator,
    AccumulatorConfig,
    Requantizer,
    RequantizerConfig,
    ShiftAdder,
    ShiftAdderConfig,
)
from neurox.macro.base import NeuroxMacroQuantMatMul
from neurox.macro.xbar_macro import XbarMacro
from neurox.mapper.xbar import (
    SerialSlicer,
    SimpleMapper,
    SimpleSlicer,
    SimpleTiler,
)
from neurox.operator import QuantSpec
from neurox.xbar import (
    Core1T1R,
    Core1T1RConfig,
    Offset1T1RXbar,
    Offset1T1RXbarConfig,
)

XbarKind = Literal["physical", "ideal"]

# dtype for the analog / solver tensors in the 1T1R path.  fp32 is
# required: the half-LSB ADC grid (first boundary ≈ 0.5 × LSB ≈ 35 µA)
# plus the large baseline-current cancellation in ref-column
# subtraction needs ≳1 µA absolute arithmetic precision — bf16's ~5 µA
# error at mA scale flips bucketize decisions near thresholds.  The
# per-cell conductance buffers are small (a few MB even at model
# scale) so the fp32/bf16 memory gap is irrelevant.
_CIRCUIT_DTYPE = torch.float32

# TOML sections with a typed ``*Config`` dataclass.  Transcoder
# sections have no dataclass (and don't need one) — the factories
# read them from the raw dict alongside the typed configs.
_XBAR1T1R_SPECS: dict[str, type] = {
    "rram": RRAMConfig,
    "nmos": NMOSConfig,
    "tia": OpAmpTIAConfig,
    "tia_nmos": NMOSConfig,
    "bl_wire": WireConfig,
    "sl_wire": WireConfig,
    "wl_wire": WireConfig,
    "sl_driver": DriverConfig,
    "wl_decoder": DecoderConfig,
    "wl_dac": GeneralDACConfig,
    "bl_adc": McsSarAdcConfig,
    "analog_mux": AnalogMuxConfig,
    "data_switchcap": SwitchCapConfig,
    "ref_switchcap": SwitchCapConfig,
    "accumulator": AccumulatorConfig,
    "shift_adder": ShiftAdderConfig,
    "core": Core1T1RConfig,
    "readout": ReadOutConfig,
    "xbar": Offset1T1RXbarConfig,
}


@cache
def _chip_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(typed, raw)`` configs for the TOML at ``config_path``.

    Cached per path so repeat calls within a run reuse the parse.
    Both forms are kept: typed dataclasses (with ``__post_init__``
    validators) for hardware modules, raw nested dict for sections
    without a dedicated typed config (``w_transcoder`` / ``x_transcoder``).
    """
    return (
        dict_configs_from_file(_XBAR1T1R_SPECS, config_path),
        dict_from_file(config_path),
    )


def derive_quant_spec(config_path: Path = DEFAULT_1T1R_TOML) -> QuantSpec:
    """Derive the operator quantisation grid from the macro's own range API.

    The macro is the source of truth for activation / weight ranges:
    ``w_value_range`` / ``x_value_range`` report the signed (symmetric)
    weight range and the unsigned activation range the hardware can
    represent.  We read them off a freshly-built physical macro
    (cheap — no fabrication) and feed them straight into
    :class:`QuantSpec``, so
    the operator grid can never drift out of sync with the macro's
    transcoder wiring.

    The **output** grid is signed and symmetric around zero — a
    linear-layer output is signed by nature (pre-activation can be
    negative).  Its width is the activation width: ``y_qmax = x_qmax``
    and ``y_qmin = -x_qmax``.  That preserves one bit of sign
    information across layers *without* introducing an extra signed-to-
    unsigned conversion hop — the next layer's asymmetric input
    quantiser re-maps into ``[x_qmin, x_qmax]`` with its own
    ``(s_x, zp_x)``, so the hardware activation grid is unchanged.
    """
    macro = _build_xbar1t1r_macro(config_path)
    x_qmin, x_qmax = macro.x_value_range
    w_qmin, w_qmax = macro.w_value_range
    # Symmetric weight is a hard macro invariant (signed-digit
    # encoding); the check catches future macro edits that forget to
    # keep this contract.
    assert w_qmin == -w_qmax, f"Expected symmetric w_value_range, got ({w_qmin}, {w_qmax})"
    return QuantSpec(x_qmin=x_qmin, x_qmax=x_qmax, w_qmax=w_qmax, y_qmin=-x_qmax, y_qmax=x_qmax)


def _build_xbar1t1r_macro(config_path: Path, *, name: str = "") -> XbarMacro:
    """Build a physical-1T1R ``XbarMacro`` from the TOML at ``config_path``.

    Builds the three-layer 1T1R stack: ``Core1T1R`` owns the array
    physics (RRAM + NMOS + OpAmpTIA + wires + solver), ``ReadOut`` performs
    the voltage-domain weighted sum, and ``Offset1T1RXbar`` glues them
    together with the offset-coding mapping.

    The leaf circuit modules (RRAM, NMOS, OpAmpTIA, wires, …) are
    still passed *as factories* one level deeper — see
    :class:`~neurox.xbar.Core1T1R` — so each core gets its own
    device instances and nothing is shared across cores
    (``temp/state_holding.md`` has the ownership rationale).  At the
    macro level, however, every component is **pre-built with its
    hierarchical profiler name** and handed to
    :class:`XbarMacro` as an instance.  The macro never threads
    naming or factory closures itself; that responsibility lives in
    this builder.

    The ``name`` argument is the hierarchical identifier the replace
    layer hands to every physical leaf (e.g. ``fc1.macro`` produces
    ``fc1.macro.xbar.core.tia`` events from the profiler).
    """
    typed, raw = _chip_config(config_path)

    rram_factory = partial(RRAM, typed["rram"], dtype=_CIRCUIT_DTYPE)
    nmos_factory = partial(NMOS, typed["nmos"], T__K=T_ROOM__K, dtype=_CIRCUIT_DTYPE)
    tia_nmos_factory = partial(NMOS, typed["tia_nmos"], T__K=T_ROOM__K, dtype=_CIRCUIT_DTYPE)
    tia_factory = partial(OpAmpTIA, typed["tia"], nmos_factory=tia_nmos_factory, dtype=_CIRCUIT_DTYPE)
    bl_wire_factory = partial(Wire, typed["bl_wire"], dtype=_CIRCUIT_DTYPE)
    sl_wire_factory = partial(Wire, typed["sl_wire"], dtype=_CIRCUIT_DTYPE)
    wl_wire_factory = partial(Wire, typed["wl_wire"], dtype=_CIRCUIT_DTYPE)

    core_factory = partial(
        Core1T1R,
        typed["core"],
        rram_factory=rram_factory,
        nmos_factory=nmos_factory,
        tia_factory=tia_factory,
        sl_wire_factory=sl_wire_factory,
        bl_wire_factory=bl_wire_factory,
        wl_wire_factory=wl_wire_factory,
        sl_driver_factory=partial(Driver, typed["sl_driver"], dtype=_CIRCUIT_DTYPE),
        wl_decoder_factory=partial(Decoder, typed["wl_decoder"]),
        wl_dac_factory=partial(GeneralDAC, typed["wl_dac"], dtype=_CIRCUIT_DTYPE),
        dtype=_CIRCUIT_DTYPE,
    )
    data_switchcap_factory = partial(SwitchCap, typed["data_switchcap"], T__K=T_ROOM__K, dtype=_CIRCUIT_DTYPE)
    ref_switchcap_factory = partial(SwitchCap, typed["ref_switchcap"], T__K=T_ROOM__K, dtype=_CIRCUIT_DTYPE)
    analog_mux_factory = partial(AnalogMux, typed["analog_mux"], dtype=_CIRCUIT_DTYPE)
    adc_factory = partial(McsSarAdc, typed["bl_adc"], T__K=T_ROOM__K, dtype=_CIRCUIT_DTYPE)

    readout_factory = partial(
        OffsetSwitchCapMuxAdcReadOut,
        typed["readout"],
        data_switchcap_factory=data_switchcap_factory,
        ref_switchcap_factory=ref_switchcap_factory,
        analog_mux_factory=analog_mux_factory,
        adc_factory=adc_factory,
        dtype=_CIRCUIT_DTYPE,
    )

    prefix = f"{name}." if name else ""
    # The unified mapper owns all static mapping strategy on its
    # sub-components: a SimpleTiler (matrix tiling), a SerialSlicer
    # (activation value decomposition), and a SimpleSlicer (weight
    # value decomposition).  The macro composes the pre-built mapper
    # with the xbar and the digital aggregation modules; it does
    # not see strategy knobs.  At call time the macro reads the
    # xbar's primitive capabilities and threads them into the
    # mapper as explicit kwargs.
    #
    # Note: the user-facing chip config still stores these slicer
    # settings under legacy ``x_transcoder`` / ``w_transcoder`` keys.
    # The runtime objects built here are slicers, not standalone
    # transcoders.
    xbar = Offset1T1RXbar(
        cfg=typed["xbar"],
        name=f"{prefix}xbar",
        core_factory=core_factory,
        readout_factory=readout_factory,
    )
    mapper = SimpleMapper(
        tiler=SimpleTiler(),
        x_slicer=SerialSlicer(
            slice_num=raw["x_transcoder"]["digit_num"],
            encoding=raw["x_transcoder"]["encoding"],
        ),
        w_slicer=SimpleSlicer(
            slice_num=raw["w_transcoder"].get("w_slice_num", 1),
            encoding=raw["w_transcoder"]["encoding"],
        ),
    )
    return XbarMacro(
        xbar=xbar,
        mapper=mapper,
        col_accumulator=Accumulator(typed["accumulator"], name=f"{prefix}col_accumulator"),
        w_shift_adder=ShiftAdder(typed["shift_adder"], name=f"{prefix}w_shift_adder"),
        x_shift_adder=ShiftAdder(typed["shift_adder"], name=f"{prefix}x_shift_adder"),
        requantizer=Requantizer(RequantizerConfig(bit_width=32), name=f"{prefix}requantizer"),
    )


def _build_xbar_ideal_macro(config_path: Path, *, name: str = "") -> XbarMacro:
    """Build the lossless twin of :func:`_build_xbar1t1r_macro`.

    Reuses the physical factory wholesale, then swaps the tile for
    an :class:`IdealXbar` via :meth:`Xbar.to_ideal` — one chip
    description, two tile implementations.  The ideal tile inherits
    the original xbar's hierarchical profiler name so reports keep
    the same identity.
    """
    macro = _build_xbar1t1r_macro(config_path, name=name)
    macro.xbar = macro.xbar.to_ideal()
    return macro


def build_macro_factory(config_path: Path, *, xbar: XbarKind) -> Callable[..., NeuroxMacroQuantMatMul]:
    """Return the per-layer macro-factory closure requested by the CLI.

    ``replace_for_hat`` / ``build_evaluator`` expect a callable that
    returns a fresh macro per layer; bind the chosen ``config_path``
    here so the CLI only deals with the two public knobs.

    The returned callable accepts ``name=<qualified layer name>`` so
    every physical leaf inside the constructed macro receives a
    hierarchical profiler identity.

    Args:
        config_path: Path to a chip-config TOML.
        xbar: ``"physical"`` → full 1T1R circuit solver;
              ``"ideal"`` → lossless reference derived from the
              physical tile.
    """
    builder = _build_xbar1t1r_macro if xbar == "physical" else _build_xbar_ideal_macro

    def factory(*, name: str = "") -> NeuroxMacroQuantMatMul:
        return builder(config_path, name=name)

    return factory

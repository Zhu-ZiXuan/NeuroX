"""CLI adapter for deterministic 1T1R cell linearization."""

from __future__ import annotations

import argparse

import torch

from neurox.tools.calibration.cli import add_calibration_args
from neurox.tools.calibration.output import emit_fragment
from neurox.tools.calibration.run import calibration_run

from .x1t1r import CalibrateCellX1t1rConfig, extract_linear_cell_config, linear_fragment_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a linear 1T1R cell from converged detailed-cell operating points"
    )
    add_calibration_args(parser, fragment=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with calibration_run(CalibrateCellX1t1rConfig, args, name="cell_linearization") as (config, output):
        result = extract_linear_cell_config(
            config.cell_config,
            v_bl_op__V=config.v_bl_op__V,
            v_sl_op__V=config.v_sl_op__V,
            v_wl_off__V=config.v_wl_off__V,
            v_wl_on__V=config.v_wl_on__V,
            device=torch.device(args.device),
            dtype={"float32": torch.float32, "float64": torch.float64}[config.dtype],
        )
        text = linear_fragment_text(result, v_bl_op__V=config.v_bl_op__V, v_sl_op__V=config.v_sl_op__V)
        emit_fragment(text, output, name="cell_linear.toml", destination=args.output)
    return 0

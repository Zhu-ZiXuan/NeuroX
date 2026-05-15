"""Stable user-facing command-line utilities for NeuroX.

Each submodule exposes a ``main()`` that is wired as a console script in
``pyproject.toml``.  After ``uv sync`` (or ``pip install -e .``) the
following commands are on ``$PATH``:

    neurox-calculate-xbar-param-1t1r   # 1T1R RRAM pre-distortion + ADC boundaries
    neurox-analyze-xbar-error-1t1r     # 1T1R VMM accuracy vs lossless ideal twin

Pass ``--help`` to any command for arguments and defaults.

Common building blocks live next to the tools:

    xbar_adc_boundries.py              # crossbar-agnostic ADC boundary helpers
"""

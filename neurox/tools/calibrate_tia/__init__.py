"""Dedicated TIA calibration: picks the :class:`OpAmpTia` internal Newton
iteration count ``n_newton`` via the shared step-ratio plateau detection plus
a relative residual guard.

CLI: ``python -m neurox.tools.calibrate_tia._opamp``

See also:
    docs/guides/calibration/README.md
"""

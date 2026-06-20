"""Calibrate a crossbar cell's internal-condensation Newton count.

A cell that condenses an internal node with a fixed unrolled Newton (for
example the access node ``V_X`` of :class:`XbarCell1T1R`) carries an
``n_newton`` knob in its own config. This package picks that count with
the same chip-parameter-free **step-ratio plateau** criterion the
array-solver calibrators use, but applied to the cell's *internal* node
and its *internal-KCL* residual rather than to the array wire / clamp
unknowns. It is a package separate from :mod:`neurox.tools.solver_calibrate`
because the internal condensation is the cell's own responsibility: a new
cell type with a different internal topology calibrates its own count
here without touching the array-solver calibrators.

The pure-Python pick logic + result containers are shared from
:mod:`neurox.tools.solver_calibrate._plateau`; this package owns only the
cell-specific operating-point grid and per-candidate sweep.

  * :mod:`._1t1r` — :class:`XbarCell1T1R` access-node condensation
    (``n_newton``).

CLI: ``python -m neurox.tools.cell_calibrate._1t1r --help``

See also:
    docs/guides/calibration/solver_iteration_counts.md
"""

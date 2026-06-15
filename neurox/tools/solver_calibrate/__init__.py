"""Tools for picking solver iteration counts via step-ratio plateau detection.

The shared methodology lives in :mod:`._plateau` (pure-Python pick logic +
result containers) and :mod:`._common` (xbar builder + workload-streaming
sweep aggregator). Each CLI module owns one solver family:

  * :mod:`.tia` — OpAmpTIA inner Newton (``n_newton``).
  * :mod:`.nested` — :class:`NestedSolver1T1R` (``n_outer`` × ``n_inner``).
  * :mod:`.full_jacobian` — :class:`FullJacobianSolver1T1R`
    (``n_newton``).

All three calibrators use the same chip-parameter-free criteria
(step-ratio plateau + relative residual guard). See
``docs/guides/calibration/README.md`` for the rationale.
"""

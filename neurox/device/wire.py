"""Metal interconnect wire model for crossbar array routing.

Physical model overview
-----------------------
Each wire is modelled as a 1-D resistor ladder: the line has ``N``
nodes (one per cell along the routed axis) plus an off-array driver,
joined by ``N`` resistive segments.  :class:`Wire` is a stateful
container that holds the per-segment equivalent resistance /
conductance tensors directly — there is no implicit "uniform
resistivity × geometric length" assumption inside the wire itself.

Why direct per-segment resistance?
----------------------------------
Real arrays route through multiple metal stacks, vias, straps, and
local widening; the equivalent resistance between two nodes is rarely
captured by a single ``r__MOhm_per_um × length`` product.  The
recommended discipline (see ``temp/wire.md``) is therefore:

* the wire's main input is a length-``N`` ``segment_r__MOhm`` tensor,
* the caller (today: :class:`~neurox.xbar.Core1T1R`) translates whatever
  high-level config it has — uniform pitch, mixed pitches, future
  strap / bridge models, etc. — into that tensor before handing it
  to :meth:`Wire.fabricate`.

The wire's interface stays narrow and topology-agnostic; new topology
parameters land on the orchestration layer, not on this module.

Segment encoding
----------------
``Wire.fabricate(segment_r__MOhm)`` takes one 1-D tensor describing
the equivalent resistance of every segment on the routed line:

* ``segment_r__MOhm[0]`` — driver to node 0.
* ``segment_r__MOhm[k]`` for ``k >= 1`` — node ``k-1`` to node ``k``.

Length = number of nodes.  The encoding handles uniform-pitch arrays
*and* non-uniform layouts without changing the solver interface.

Owned buffers
-------------
After ``fabricate``, the wire holds two non-persistent buffers, both
length ``N``:

* ``segment_r__MOhm`` — per-segment equivalent resistance [MOhm].
* ``segment_g__uS``   — per-segment equivalent conductance [uS].

Solver-specific Jacobian templates (the tridiagonal diagonal /
off-diagonal) are **not** cached here.  They are derived inside the
solver from ``segment_g__uS`` so the Wire stays a pure static
discretization holder, free of any particular Newton-iteration
representation.

Capacitance still lives on the config as a lumped
``c__fF_per_um``-per-length parameter.  The energy path multiplies it
by a geometric total length read from :class:`Core1T1RConfig`'s
spacings — a deliberate short-term simplification (see
``temp/wire.md``); per-segment capacitance can be added later
without touching the resistive-solver path.

Scope
-----
The current implementation only models a 1-D ladder.  Future
strap / bridge / comb source-line networks will live in a separate
network module rather than expanding this class — keeping
:class:`Wire` topology-agnostic protects the solver's
tridiagonal-Jacobian fast path.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class WireConfig:
    """Immutable per-length capacitance configuration for a metal wire.

    Resistance is *not* configured here: each :class:`Wire` instance
    receives a direct per-segment resistance tensor at
    :meth:`Wire.fabricate` time, so the wire never assumes a uniform
    resistivity.  See ``temp/wire.md``.

    Attributes:
        c__fF_per_um: Capacitance per unit length [fF/um].  Must be
            strictly positive.  Consumed only by the energy path
            (``Core1T1R._compute_array_energy__fJ``); the solver does
            not read it.
    """

    c__fF_per_um: float

    def __post_init__(self) -> None:
        if not (self.c__fF_per_um > 0.0):
            raise ValueError(f"require: c__fF_per_um ({self.c__fF_per_um}) > 0.0")


class Wire(nn.Module):
    """Stateful 1-D interconnect-wire model with direct per-segment resistance.

    Each :class:`Wire` instance internally owns the discretized
    per-segment resistance / conductance tensors consumed by the 1T1R
    Newton solver.  Lifecycle:

    1. ``__init__(cfg)`` — bind the per-length capacitance constant;
       register empty placeholder buffers.
    2. ``fabricate(segment_r__MOhm)`` — accept a length-``N`` direct-
       resistance tensor (``segment_r__MOhm[0]`` = driver-to-first;
       ``segment_r__MOhm[k>=1]`` = node ``k-1`` → ``k``), register the
       resistance buffer, and derive the per-segment conductance.
       Re-callable: a subsequent ``fabricate`` replaces every buffer
       with a fresh sample.

    The class is intentionally *stateful but solver-free*: the wire
    does **not** provide ``solve_dc``, and it does **not** cache any
    tridiagonal Jacobian templates.  Cell injection and boundary
    clamp are coupled to the line equations through the global
    :class:`~neurox.xbar.NewtonRaphsonSolver1T1R`, which
    builds its Jacobian templates locally from ``segment_g__uS``.

    Attributes:
        cfg: Per-length capacitance :class:`WireConfig`.
        dtype: Tensor dtype for every fabricated buffer.

    Registered buffers (non-persistent, both shape ``(N,)``):
        segment_r__MOhm: Per-segment equivalent resistance [MOhm].
        segment_g__uS:   Per-segment equivalent conductance [uS].
    """

    segment_r__MOhm: Tensor
    segment_g__uS: Tensor

    def __init__(self, cfg: WireConfig, *, name: str = "", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self._neurox_name = name
        self.cfg = cfg
        self.dtype = dtype
        for buf_name in ("segment_r__MOhm", "segment_g__uS"):
            self.register_buffer(buf_name, torch.empty(0, dtype=dtype), persistent=False)

    @property
    def c__fF_per_um(self) -> float:
        return self.cfg.c__fF_per_um

    def fabricate(self, segment_r__MOhm: Tensor) -> None:
        """Bind a per-segment resistance tensor and derive the conductance.

        Args:
            segment_r__MOhm: Equivalent resistance of every segment
                on the routed line [MOhm].  1-D tensor of length
                ``N`` (= number of nodes).  ``segment_r__MOhm[0]`` is
                the driver-to-node-0 segment; ``segment_r__MOhm[k]``
                for ``k >= 1`` is the node-``(k-1)`` → node-``k``
                segment.  Cast to ``self.dtype`` and moved onto the
                wire's device on ingest.
        """
        segment_r = segment_r__MOhm.to(dtype=self.dtype, device=self.segment_r__MOhm.device)
        if segment_r.ndim != 1:
            raise ValueError(f"require: segment_r__MOhm.ndim ({segment_r.ndim}) == 1")
        num_nodes = segment_r.numel()
        if not (num_nodes >= 1):
            raise ValueError(f"require: segment_r__MOhm.numel ({num_nodes}) >= 1")
        if not bool((segment_r > 0.0).all()):
            raise ValueError(f"require: segment_r__MOhm > 0.0 (got min {segment_r.min().item()})")

        segment_g = 1.0 / segment_r

        self.register_buffer("segment_r__MOhm", segment_r, persistent=False)
        self.register_buffer("segment_g__uS", segment_g, persistent=False)

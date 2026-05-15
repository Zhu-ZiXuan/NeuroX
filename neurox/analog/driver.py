"""Ideal constant-voltage clamp-driver model.

A :class:`Driver` is the **ideal** concrete implementation of the
:class:`~neurox.analog.ClampDriver` protocol: a fixed nominal voltage
``drive_value`` [V] with optional per-element Gaussian thermal noise.
The 1T1R solver consumes it through the protocol's structural surface
— :attr:`v_ref__V`, :meth:`fabricate`, :meth:`snapshot`,
:meth:`solve_clamp` — without ever asking for a special open-loop
"give me a voltage tensor" shortcut.

API surface
-----------
* :meth:`fabricate(shape)` — placeholder for future per-instance
  mismatch sampling; no-op today.
* :meth:`snapshot(shape=...)` — sample a noisy clamp voltage once per
  VMM, bundled into a :class:`DriverSnapshot`.
* :meth:`solve_clamp(i_port__uA, snapshot, v_clamp_init__V=None)` —
  protocol-facing entry point.  Returns the snapshot's clamp voltage
  expanded to the port-current shape and a zero-tensor input
  impedance.  The ideal driver has no feedback, so the boundary's
  port-output current never changes the clamp.
* :meth:`solve_dc(...)` — richer concrete entry point returning a
  :class:`DriverDC`.  Same physical answer; carries the named
  dataclass for callers that prefer the typed return.

Energy / lifecycle alignment
----------------------------
:class:`Driver` is treated as a *boundary model* on the same footing
as :class:`~neurox.analog.opamp_tia.OpAmpTIA` — it does not report a separate
per-VMM dynamic energy.  Joule dissipation through the boundary is
already captured by the solver's supply-power term in
``Core1T1R._compute_array_energy__fJ`` (``v_sl_drive · i_sl_driver``);
counting an extra ``energy_per_op`` term on the driver itself would
double-book the same charge.  Static PPA (``area_per_inst__um2``,
``leakage_per_inst__uW``, ``latency_per_op__ns``) is kept.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian


@dataclass(frozen=True)
class DriverConfig:
    """Immutable configuration for an ideal constant-voltage clamp driver.

    Attributes:
        drive_value: Nominal output voltage in [V].
        drive_thermal: Gaussian thermal noise added to each driven sample
            (sigma in [V]); models output-stage thermal variation.
            ``None`` skips this noise stage.
        latency_per_op__ns: Drive latency per operation in [ns].
        leakage_per_inst__uW: Static leakage power per driver instance in [uW].
        area_per_inst__um2: Silicon area per driver instance in [um^2].
    """

    drive_value: float

    drive_thermal: float | None = None

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0


@dataclass(frozen=True)
class DriverSnapshot:
    """Per-VMM snapshot for an ideal :class:`Driver`-as-:class:`ClampDriver`.

    Carries the noisy clamp-voltage sample taken once per VMM by
    :meth:`Driver.snapshot`; threaded through every
    :meth:`Driver.solve_dc` / :meth:`Driver.solve_clamp` call so the
    Newton iteration and any post-solver re-call see an identical
    fixed sample of the output-stage thermal noise.

    Attributes:
        v_clamp__V: Noisy nominal output voltage [V], broadcast to the
            execution port shape.
    """

    v_clamp__V: Tensor


@dataclass(frozen=True)
class DriverDC:
    """Richer DC result of an ideal :class:`Driver` clamp eval.

    Returned by :meth:`Driver.solve_dc` for callers that prefer a
    named dataclass over the bare tuple.  Ideal drivers report a fixed
    (snapshot-sampled) ``v_clamp__V`` and zero input impedance, since
    the boundary acts as a static voltage source once the per-VMM
    noise sample is locked in.  The solver-facing
    :meth:`Driver.solve_clamp` returns the same two tensors as a plain
    ``tuple[Tensor, Tensor]`` to match the :class:`ClampDriver`
    protocol.

    Attributes:
        v_clamp__V: Boundary clamp voltage [V] for this VMM.
        dVclamp_dI__MOhm: Boundary input impedance [MOhm].  Always
            zero for an ideal driver.
    """

    v_clamp__V: Tensor
    dVclamp_dI__MOhm: Tensor


class Driver(nn.Module):
    """Constant-voltage clamp driver with Gaussian thermal noise.

    Implements the :class:`~neurox.analog.ClampDriver` protocol — the
    1T1R solver uses one of these on each clamp port (BL or SL) when
    the boundary is an ideal voltage source.

    Registered non-persistent buffer:
        ``nominal_drive_value`` — 0-d scalar at ``cfg.drive_value``,
            built once and never overwritten.  Per the project-wide
            ``fabricate`` contract :meth:`fabricate(shape)` is a no-op
            today; output-stage randomness is sampled afresh inside
            :meth:`snapshot` from the nominal scalar on every
            call.

    Args:
        cfg: Immutable driver configuration.
        dtype: Floating-point dtype for the ``nominal_drive_value``
            buffer.
    """

    nominal_drive_value: Tensor

    def __init__(self, cfg: DriverConfig, *, name: str = "", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg

        self.register_buffer(
            "nominal_drive_value",
            torch.tensor(cfg.drive_value, dtype=dtype),
            persistent=False,
        )

    @property
    def v_ref__V(self) -> float:
        """Ideal / zero-current clamp voltage [V] — ``ClampDriver`` protocol surface.

        For the ideal constant-voltage driver this is simply the
        nominal ``drive_value``: the boundary's static operating
        point in the absence of thermal noise.
        """
        return self.cfg.drive_value

    @property
    def area_per_inst__um2(self) -> float:
        """Circuit area per instance in [um2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Circuit leakage power per instance in [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per operation in [ns]."""
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """No-op fabricate — ideal driver has no static mismatch state.

        Kept for lifecycle consistency with the other circuit modules
        (per ``temp/fabricate.md``).  Output-stage randomness is
        sampled afresh inside :meth:`snapshot` from the nominal
        every VMM rather than stamped at fabrication time, so this
        call has nothing to do.

        Args:
            shape: Reserved for future per-instance static mismatch.
                Ignored today.
        """
        return

    # --- ClampDriver protocol ---

    def snapshot(self, *, shape: tuple[int, ...]) -> DriverSnapshot:
        """Return a per-VMM snapshot for the ideal clamp driver path.

        Samples the noisy clamp voltage once by cloning
        ``nominal_drive_value`` (clone-before-expand decouples the
        sampled tensor's storage from the nominal buffer), expanding
        to ``shape``, and applying the optional Gaussian thermal
        noise.  The same snapshot is reused across every outer
        Newton iteration and any post-solver re-call, so the
        boundary sees a fixed noise sample for the whole VMM (per
        ``temp/state_holding.md``).

        Args:
            shape: Port shape ``(*batch, port_num)`` for the clamp
                sample.

        Returns:
            :class:`DriverSnapshot` with the sampled ``v_clamp__V``.
        """
        v_clamp__V = self.nominal_drive_value.clone().expand(shape)
        if self.cfg.drive_thermal is not None:
            v_clamp__V = apply_gaussian(v_clamp__V, self.cfg.drive_thermal)
        return DriverSnapshot(v_clamp__V=v_clamp__V)

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snapshot: DriverSnapshot,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> DriverDC:
        """Return the ideal-clamp working point at the present port current.

        An ideal voltage-source clamp's output is insensitive to the
        port current — the snapshot's pre-sampled ``v_clamp__V`` is
        returned unchanged, and the boundary's input impedance is
        identically zero.

        Args:
            i_port__uA: Port-output current [uA] — positive when
                current leaves the clamp port (driver is sourcing),
                negative when it enters the port (sinking).  Ignored
                by the ideal driver other than as the broadcast
                shape for the returned tensors.
            snapshot: Per-VMM snapshot from :meth:`snapshot`; carries
                the noisy clamp voltage.
            v_clamp_init__V: Optional warm-start hint from the solver.
                Ignored — the ideal driver does not run a local Newton
                iteration.

        Returns:
            :class:`DriverDC` carrying the snapshot voltage and a
            zero-tensor input impedance broadcast to the
            ``i_port__uA`` shape.
        """
        v_clamp__V = snapshot.v_clamp__V.expand_as(i_port__uA)
        dVclamp_dI__MOhm = torch.zeros_like(i_port__uA)
        return DriverDC(v_clamp__V=v_clamp__V, dVclamp_dI__MOhm=dVclamp_dI__MOhm)

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: DriverSnapshot,
        *,
        v_clamp_init__V: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Solver-facing :class:`ClampDriver` wrapper around :meth:`solve_dc`.

        Returns the minimal pair the wire Newton step needs.  The
        richer :class:`DriverDC` from :meth:`solve_dc` is unpacked
        into a plain tuple so the
        :class:`~neurox.analog.ClampDriver` Protocol can express the
        surface as a fixed ``tuple[Tensor, Tensor]`` without leaking
        the concrete result dataclass through the structural
        Protocol.

        Args:
            i_port__uA: Port-output current [uA]; see
                :meth:`solve_dc` for the sign convention.  Forwarded
                verbatim to :meth:`solve_dc`.
            snapshot: Per-VMM snapshot from :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint forwarded
                verbatim (ignored by the ideal driver).

        Returns:
            ``(v_clamp__V, dVclamp_dI__MOhm)`` — same two tensors that
            :meth:`solve_dc` would have packaged into a
            :class:`DriverDC`.
        """
        dc = self.solve_dc(i_port__uA, snapshot, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm

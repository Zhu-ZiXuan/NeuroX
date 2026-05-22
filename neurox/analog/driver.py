"""Ideal constant-voltage clamp-driver model."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ValidateMixin
from neurox.common.nonideality import apply_gaussian


@dataclass(frozen=True)
class DriverConfig(ValidateMixin):
    """Configuration for an ideal constant-voltage clamp driver.

    Attributes:
        drive_value: Ideal clamp voltage [V].
        drive_thermal__V: Per-solve Gaussian thermal noise sigma [V].
        enable_drive_thermal: Apply ``drive_thermal__V`` at snapshot time.
        latency_per_op__ns: Latency per operation [ns].
        leakage_per_inst__uW: Leakage power per instance [uW].
        area_per_inst__um2: Area per instance [um^2].
    """

    # --- Drive ---
    drive_value: float

    # --- Drive thermal noise ---
    drive_thermal__V: float
    enable_drive_thermal: bool

    # --- PPA ---
    latency_per_op__ns: float
    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_drive()
        self.validate_ppa()

    def validate_drive(self) -> None:
        self._require_nonneg(self.drive_thermal__V, "drive_thermal__V")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class DriverSnapshot:
    """One sampled driver snapshot.

    Attributes:
        v_clamp__V: Sampled clamp voltage [V].
    """

    v_clamp__V: Tensor


@dataclass(frozen=True)
class DriverDCOP:
    """DC result of one ideal clamp evaluation.

    Attributes:
        v_clamp__V: Clamp voltage [V].
        dVclamp_dI__MOhm: Small-signal slope dV/dI [MOhm].
    """

    v_clamp__V: Tensor
    dVclamp_dI__MOhm: Tensor


class Driver(FabricateMixin, nn.Module):
    """Constant-voltage clamp driver with optional thermal noise."""

    nominal_drive_value: Tensor

    def __init__(
        self,
        *,
        cfg: DriverConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Construct one ideal clamp driver.

        Args:
            cfg: Concrete configuration dataclass.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
        """
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg
        self._inst_shape = inst_shape
        self.dtype = dtype
        self.T__K = T__K

        self.register_buffer(
            "nominal_drive_value",
            torch.tensor(cfg.drive_value, dtype=dtype),
            persistent=False,
        )

    @property
    def v_ref__V(self) -> float:
        """Ideal / zero-current clamp voltage [V]."""
        return self.cfg.drive_value

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Latency per op [ns]."""
        return self.cfg.latency_per_op__ns

    # --- ClampDriver protocol ---

    def snapshot(self, *, shape: tuple[int, ...]) -> DriverSnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Snapshot shape.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        v_clamp__V = apply_gaussian(
            self.nominal_drive_value.clone().expand(shape),
            self.cfg.drive_thermal__V,
            enabled=self.cfg.enable_drive_thermal,
        )
        return DriverSnapshot(v_clamp__V=v_clamp__V)

    def solve_dc(
        self,
        i_port__uA: Tensor,
        snapshot: DriverSnapshot,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> DriverDCOP:
        """Solve the ideal clamp at the present port current.

        Args:
            i_port__uA: Port current [uA].
            snapshot: Snapshot returned by :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint [V]. Ignored by the ideal
                driver.

        Returns:
            Clamp voltage and small-signal slope at the requested operating
            point.
        """
        v_clamp__V = snapshot.v_clamp__V.expand_as(i_port__uA)
        dVclamp_dI__MOhm = torch.zeros_like(i_port__uA)
        return DriverDCOP(v_clamp__V=v_clamp__V, dVclamp_dI__MOhm=dVclamp_dI__MOhm)

    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: DriverSnapshot,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return the tuple-based clamp result required by the solver.

        Args:
            i_port__uA: Port current [uA].
            snapshot: Snapshot returned by :meth:`snapshot`.
            v_clamp_init__V: Optional warm-start hint [V]. Ignored by the ideal
                driver.

        Returns:
            Tuple `(v_clamp__V, dVclamp_dI__MOhm)`.
        """
        dc = self.solve_dc(i_port__uA, snapshot, v_clamp_init__V=v_clamp_init__V)
        return dc.v_clamp__V, dc.dVclamp_dI__MOhm

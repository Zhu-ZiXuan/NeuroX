"""Ideal constant-voltage clamp-driver model."""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.common.nonideality import apply_gaussian


@dataclass(frozen=True)
class DriverConfig(CircuitConfig):
    """Configuration for an ideal constant-voltage clamp driver.

    Attributes:
        drive_value: Ideal clamp voltage [V].
        drive_thermal__V: Per-solve Gaussian thermal noise sigma [V].
    """

    # --- Drive ---
    drive_value: float

    # --- Drive thermal noise ---
    drive_thermal__V: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_drive()
        self.validate_ppa()

    def validate_drive(self) -> None:
        self._require_nonneg(self.drive_thermal__V, "drive_thermal__V")


@dataclass(frozen=True)
class DriverPolicy:
    """Per-source toggles selecting which driver nonidealities are active.

    Attributes:
        drive_thermal: Apply ``drive_thermal__V`` at snapshot time.
    """

    drive_thermal: bool


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


class Driver(CircuitBase[DriverConfig]):
    """Constant-voltage clamp driver with optional thermal noise."""

    nominal_drive_value: Tensor

    def __init__(
        self,
        *,
        config: DriverConfig,
        policy: DriverPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Construct one ideal clamp driver.

        Args:
            config: Concrete configuration dataclass.
            policy: Per-source nonideality enable flags.
            name: Hierarchical instance name used by the profiler.
            inst_shape: Per-instance fabrication shape.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature [K].
        """
        super().__init__(config=config, name=name, inst_shape=inst_shape)

        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K

        self.register_buffer(
            "nominal_drive_value",
            torch.tensor(config.drive_value, dtype=dtype),
            persistent=False,
        )

    @property
    def v_ref__V(self) -> float:
        """Ideal / zero-current clamp voltage [V]."""
        return self.config.drive_value

    # --- Snapshot + clamp solve ---

    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> DriverSnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snapshot fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        v_view = self.nominal_drive_value.expand(shape) if shape else self.nominal_drive_value
        v = v_view if multi_coords is None else v_view[multi_coords]
        v = apply_gaussian(
            v.clone(),
            self.config.drive_thermal__V,
            enabled=self.policy.drive_thermal,
        )
        return DriverSnapshot(v_clamp__V=v)

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

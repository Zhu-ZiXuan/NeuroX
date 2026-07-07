"""Switched-capacitor bank for passive charge sharing.

See also:
    docs/reference/analog/switch_cap.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.common.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K


@dataclass(frozen=True, kw_only=True)
class SwitchCapConfig(CircuitConfig):
    """Immutable physical configuration for :class:`SwitchCap`.

    Attributes:
        c_unit__fF: Unit capacitance.
        cap_mismatch_sigma_relative: Per-unit-cap Pelgrom relative σ.
        energy_per_sample_overhead__fJ: Per-bank switching overhead.
        latency_per_op__ns: Per-sample-and-accumulate latency;
            multiplied by the runtime serial-op count at logging time.
    """

    # --- Unit capacitance ---
    c_unit__fF: float

    # --- Cap mismatch (Pelgrom) ---
    cap_mismatch_sigma_relative: float

    # --- Energy / latency ---
    energy_per_sample_overhead__fJ: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_capacitance()
        self.validate_noise()
        self.validate_ppa()

    def validate_capacitance(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")

    def validate_noise(self) -> None:
        self._require_nonneg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_ppa(self) -> None:
        super().validate_ppa()
        self._require_nonneg(self.energy_per_sample_overhead__fJ, "energy_per_sample_overhead__fJ")
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class SwitchCapPolicy:
    """Per-source toggles selecting which SwitchCap nonidealities are active.

    Attributes:
        cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at fabricate time.
        sampling_thermal_noise: Apply kT/C settling noise at sample time.
    """

    cap_mismatch: bool
    sampling_thermal_noise: bool


class SwitchCap(CircuitBase[SwitchCapConfig]):
    """Bottom-plate-sampled cap bank with passive charge-share averaging.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        cap_weights: Per-cap multipliers on ``config.c_unit__fF``.
    """

    nominal_c__fF: Tensor
    c__fF: Tensor

    def __init__(
        self,
        *,
        config: SwitchCapConfig,
        policy: SwitchCapPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        cap_weights: tuple[float, ...],
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        if not (T__K > 0.0):
            raise ValueError(f"SwitchCap.T__K ({T__K}) must be > 0")
        if len(cap_weights) < 1:
            raise ValueError(f"require: len(cap_weights) ({len(cap_weights)}) >= 1")
        for k, w in enumerate(cap_weights):
            if not (w > 0.0):
                raise ValueError(f"require: cap_weights[{k}] ({w}) > 0")

        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype
        self.n_caps = len(cap_weights)

        nominal_c__fF = config.c_unit__fF * torch.tensor(cap_weights, dtype=dtype)
        self.register_buffer("nominal_c__fF", nominal_c__fF, persistent=False)
        self.register_buffer(
            "c__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """Resample per-cap mismatch at ``(*self._inst_shape, n_caps)``."""
        config = self.config
        self.c__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*self._inst_shape, self.n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=self.policy.cap_mismatch,
        )

    def sample_and_accumulate(self, v_in__V: Tensor) -> Tensor:
        """Sample per-cap voltages and run passive charge-sharing.

        Args:
            v_in__V: Per-cap sampled voltages,
                shape ``(*batch, *bank_shape, n_caps)``.

        Returns:
            Node voltage with shape ``(*batch, *bank_shape)``.
        """
        c__fF = self.c__fF
        # kT/C settling noise: kt__fJ = k_B·T·1e15 so kt/c lands in V^2.
        kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
        sigma__V = torch.sqrt(kt__fJ / c__fF)
        v_hold__V = apply_gaussian(v_in__V, sigma__V, enabled=self.policy.sampling_thermal_noise)
        # Passive charge-share: node settles to the charge-weighted mean of the
        # held voltages.
        c_total__fF = c__fF.sum(dim=-1)
        v_out__V = torch.sum(c__fF * v_hold__V, dim=-1) / c_total__fF

        # Serial op count via the position-invariant numel rule on the
        # output ``v_out__V`` (n_caps was already reduced out, so divisor
        # is just inst_count — same form as every other emitting leaf).
        e_caps__fJ = 0.5 * torch.sum(c__fF * v_in__V * v_in__V, dim=-1)
        dynamic_energy__fJ = e_caps__fJ + self.config.energy_per_sample_overhead__fJ
        serial_op_count = max(1, v_out__V.numel() // max(self.inst_count, 1))
        latency__ns = torch.tensor(
            self.config.latency_per_op__ns * serial_op_count,
            device=v_in__V.device,
            dtype=dynamic_energy__fJ.dtype,
        )
        self._log_dynamic_energy(dynamic_energy__fJ)
        self._log_latency(latency__ns)
        return v_out__V

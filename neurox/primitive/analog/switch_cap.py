"""Switched-capacitor bank for passive charge sharing.

See also:
    docs/reference/primitive/analog/switch_cap.md
"""

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy
from neurox.primitive.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.primitive.physical_constant import K_BOLTZMANN__J_per_K


class SwitchCapConfig(AnalogConfig):
    """Immutable physical configuration for :class:`SwitchCap`.

    Attributes:
        c_unit__fF: Unit capacitance.
        cap_mismatch_sigma_relative: Per-unit-cap Pelgrom relative σ.
        energy_per_sample_overhead__fJ: Per-bank switching overhead.
        latency_per_op__ns: Per-sample-and-accumulate latency;
            multiplied by the runtime serial-op count at logging time.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    c_unit__fF: float
    cap_mismatch_sigma_relative: float
    energy_per_sample_overhead__fJ: float
    latency_per_op__ns: float
    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def validate(self) -> None:
        self.validate_capacitance()
        self.validate_noise()
        self.validate_ppa()

    def validate_capacitance(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")

    def validate_noise(self) -> None:
        self._require_non_neg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_non_neg(self.energy_per_sample_overhead__fJ, "energy_per_sample_overhead__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class SwitchCapPolicy(AnalogPolicy):
    """Per-source toggles selecting which SwitchCap nonidealities are active.

    Attributes:
        cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at fabricate time.
        sampling_thermal_noise: Apply kT/C settling noise at sample time.
    """

    cap_mismatch: bool
    sampling_thermal_noise: bool


class SwitchCap(AnalogBase[SwitchCapConfig, SwitchCapPolicy]):
    """Bottom-plate-sampled cap bank with passive charge-share averaging.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        cap_weights: Per-cap multipliers on ``config.c_unit__fF``.
    """

    # --- Fabrication source buffers ---

    nominal_c__fF: Tensor

    def __init__(
        self,
        *,
        config: SwitchCapConfig,
        policy: SwitchCapPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        cap_weights: tuple[float, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        if not (T__K > 0.0):
            raise ValueError(f"SwitchCap.T__K ({T__K}) must be > 0")
        if len(cap_weights) < 1:
            raise ValueError(f"require: len(cap_weights) ({len(cap_weights)}) >= 1")
        for k, w in enumerate(cap_weights):
            if not (w > 0.0):
                raise ValueError(f"require: cap_weights[{k}] ({w}) > 0")

        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
        self.T__K = T__K
        self.n_caps = len(cap_weights)
        self._register_fabrication_buffers(dtype=dtype, cap_weights=cap_weights)

    def _register_fabrication_buffers(
        self,
        *,
        dtype: torch.dtype,
        cap_weights: tuple[float, ...],
    ) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "nominal_c__fF",
            self.config.c_unit__fF * torch.tensor(cap_weights, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        config = self.config
        self.c__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*self.inst_shape, self.n_caps),
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

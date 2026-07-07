"""Ideal current multiplexer — single-ended N:1 time-share transport block.

See also:
    docs/reference/analog/current_mux.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig


@dataclass(frozen=True, kw_only=True)
class CurrentMuxConfig(CircuitConfig):
    """Immutable configuration for :class:`CurrentMux`.

    Attributes:
        select_num: Design N of the N:1 fan-in — the number of columns
            sharing one lane. Cross-checked by the caller against its
            reference group size; it does NOT scale energy or latency.
        mux_gain: Scalar matched transport gain (copy/transport factor).
        v_supply__V: Rail supply voltage driving the data-dependent
            transport dissipation.
        latency_per_op__ns: Per-transport latency; multiplied by the
            runtime serial-op count at logging time.
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    # --- Fan-in (design only) ---
    select_num: int

    # --- Gain ---
    mux_gain: float

    # --- Rail ---
    v_supply__V: float

    # --- Latency ---
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_fan_in()
        self.validate_gain()
        self.validate_rail()
        self.validate_latency()
        self.validate_ppa()

    def validate_fan_in(self) -> None:
        self._require_pos(self.select_num, "select_num")

    def validate_gain(self) -> None:
        self._require_pos(self.mux_gain, "mux_gain")

    def validate_rail(self) -> None:
        self._require_pos(self.v_supply__V, "v_supply__V")

    def validate_latency(self) -> None:
        self._require_nonneg(self.latency_per_op__ns, "latency_per_op__ns")


@dataclass(frozen=True)
class CurrentMuxPolicy:
    """Abstract marker for CurrentMux nonideality policy — no sources."""


class CurrentMux(CircuitBase[CurrentMuxConfig]):
    """Ideal N:1 time-share current mux — identity·gain transport.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        read_pulse__ns: Read-window width passed by the caller;
            scales the per-call rail energy.
    """

    def __init__(
        self,
        *,
        config: CurrentMuxConfig,
        policy: CurrentMuxPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        read_pulse__ns: float,
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K
        self.read_pulse__ns = read_pulse__ns

    def _sample_fabricate_mismatch(self) -> None:
        pass  # ideal identity-gain transport: no static mismatch

    def transport(self, i__uA: Tensor) -> Tensor:
        """Transport one current through the shared lane at the configured gain.

        Args:
            i__uA: Per-column input current.

        Returns:
            Lane output current ``mux_gain * i__uA``.
        """
        i_out__uA = self.config.mux_gain * i__uA

        # Rail dissipation on the output lane only: V_supply·|i_out|·t. The
        # input current is sourced externally (its production energy is
        # accounted by the upstream block), so it is NOT counted here.
        # uA·V·ns = fJ.
        dynamic_energy__fJ = self.config.v_supply__V * i_out__uA.abs() * self.read_pulse__ns
        self._log_dynamic_energy(dynamic_energy__fJ)

        # serial_op_count counts the per-group column visits already; the
        # N:1 fan-in (select_num) is NOT an extra multiplier.
        if self.config.latency_per_op__ns > 0.0:
            serial_op_count = max(1, i__uA.numel() // max(self.inst_count, 1))
            latency__ns = torch.tensor(
                self.config.latency_per_op__ns * serial_op_count,
                device=i__uA.device,
                dtype=dynamic_energy__fJ.dtype,
            )
            self._log_latency(latency__ns)
        return i_out__uA

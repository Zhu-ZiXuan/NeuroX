"""Digital-to-analog converter (DAC) models for CiM macro simulations.

A DAC maps integer input codes to analog voltage levels using a fixed
code-to-signal lookup table, then adds per-output Gaussian thermal noise
to model output-driver imprecision.

The ``convert(code)`` method performs:
1. LUT indexing: ``signal = code_to_signal[code]`` (float voltage in [V]).
2. Additive Gaussian thermal noise (``drive_thermal``) on each output sample.
3. Returns ``(signal, dynamic_energy__fJ)`` where the energy is a scalar
   proportional to the number of converted samples.

``GeneralDAC`` is the concrete implementation; ``GeneralDACConfig`` holds the
LUT, noise config, and cost metrics.  ``DAC`` is the abstract base.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.profiler import ProfiledModule


class DAC(ABC):
    """Abstract base class for all DAC models.

    Concrete subclasses must implement:
    - ``convert(code)``: map integer codes to analog signals.  Dynamic
      energy and latency are emitted through the profiler side channel.
    - ``code_to_signal``: the code → nominal-voltage LUT (a ``Tensor``
      of length ``N_codes``).  Consumers like :class:`Xbar1T1R`
      directly index this to resolve operating-point voltages (e.g.
      the WL-on level at ``code_to_signal[1]`` for a 2-state WL DAC).
    - Cost-metric properties: ``area_per_inst__um2``, ``leakage_per_inst__uW``,
      ``latency_per_op__ns``.
    """

    # Attribute annotation: every concrete DAC implementation must
    # expose its LUT under this name (typically via ``register_buffer``
    # in ``__init__``).  The annotation here makes static type
    # checkers accept direct access from callers that hold only the
    # abstract ``DAC`` handle.
    code_to_signal: Tensor

    @property
    @abstractmethod
    def area_per_inst__um2(self) -> float:
        """Circuit area per instance in [um2]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def leakage_per_inst__uW(self) -> float:
        """Circuit leakage power per instance in [uW]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def latency_per_op__ns(self) -> float:
        """Latency per operation in [ns]."""
        raise NotImplementedError

    @abstractmethod
    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog signals.

        Args:
            code: Integer input codes. Shape: arbitrary.

        Returns:
            Float analog signal of the same shape as ``code``.
        """
        raise NotImplementedError

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Default no-op — DAC has no static fabrication state today.

        The unified ``fabricate(shape)`` signature is kept for
        lifecycle consistency with the other readout-chain modules
        (per ``temp/fabricate.md``).  A future iteration may introduce
        static per-output mismatch (LUT-entry offset, gain spread)
        sized over ``shape``; until then the call has nothing to do.

        Args:
            shape: Reserved for future per-output static mismatch.
                Ignored today.
        """
        return


@dataclass(frozen=True)
class GeneralDACConfig:
    """Immutable configuration for ``GeneralDAC``.

    Attributes:
        code_to_signal: Voltage lookup table indexed by integer code.
            ``code_to_signal[i]`` is the nominal analog output in [V] for
            digital code ``i``.  Length equals the number of input codes.
        drive_thermal: Gaussian thermal noise added to each output sample
            after LUT lookup (sigma in [V]).  ``None`` skips this noise.
        energy_per_op__fJ: Dynamic energy per conversion operation in [fJ].
        latency_per_op__ns: Conversion latency per operation in [ns].
        leakage_per_inst__uW: Static leakage power per DAC instance in [uW].
        area_per_inst__um2: Silicon area per DAC instance in [um^2].
    """

    code_to_signal: list[float]

    drive_thermal: float | None = None

    energy_per_op__fJ: float = 0.0

    latency_per_op__ns: float = 0.0
    leakage_per_inst__uW: float = 0.0
    area_per_inst__um2: float = 0.0


class GeneralDAC(nn.Module, DAC, ProfiledModule):
    """Configurable DAC model with LUT-based conversion and thermal noise.

    Maps integer codes to analog voltages via a registered LUT buffer, then
    perturbs each output with Gaussian thermal noise (``config.drive_thermal``).

    Registered non-persistent buffer: ``code_to_signal`` (float, [N_codes]).

    Args:
        cfg: Immutable DAC configuration.
        name: Hierarchical profiler name (e.g. ``<core>.wl_dac``).
        dtype: Floating-point dtype used for the LUT buffer and output signal.
    """

    code_to_signal: Tensor

    def __init__(self, cfg: GeneralDACConfig, *, name: str = "", dtype: torch.dtype = torch.float32) -> None:
        """Initialize general DAC.

        Args:
            cfg: DAC configuration.
            name: Hierarchical profiler name.
            dtype: Float dtype for output signal.
        """
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)

        self.cfg = cfg

        self.register_buffer("code_to_signal", torch.tensor(cfg.code_to_signal, dtype=dtype), persistent=False)

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

    def convert(self, code: Tensor) -> Tensor:
        """Convert integer digital codes to float analog voltages.

        Indexes ``code_to_signal`` LUT by ``code``, then adds per-element
        Gaussian thermal noise (``cfg.drive_thermal``).  Dynamic energy
        and latency are emitted through the profiler side channel.

        Args:
            code: Integer input codes. Shape: arbitrary.

        Returns:
            Float analog voltages [V] of the same shape as ``code``.
        """
        # --- map code to analog level ---
        signal = self.code_to_signal[code]

        # --- apply drive noise ---
        if self.cfg.drive_thermal is not None:
            signal = apply_gaussian(signal, self.cfg.drive_thermal)

        # --- dynamic energy emitted as side-channel event ---
        if self.cfg.energy_per_op__fJ != 0.0:
            dynamic_energy__fJ = torch.full_like(signal, self.cfg.energy_per_op__fJ, dtype=torch.float32)
            self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        else:
            self._log_dynamic(0.0, self.cfg.latency_per_op__ns)

        return signal

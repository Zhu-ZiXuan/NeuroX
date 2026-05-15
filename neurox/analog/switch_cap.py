"""Switched-capacitor bank — passive charge-sharing averager.

Position in the signal chain::

    core_1t1r  ->  readout  ->  analog_mux  ->  differential ADC

``SwitchCap`` is a reusable analog block specialised for the xbar
readout path.  One instance models a bank of bottom-plate-sampled
capacitors.  In the offset-coded 1T1R path one bank serves one
logical data, the per-cap *count* is the upstream ``w_digit_count``,
and the per-cap *sizes* set the cap ratios driving the passive
charge-share average.

Physical model — passive charge-sharing
---------------------------------------
The bank performs **bottom-plate sampling followed by passive charge
sharing** at the top-plate node.  Each cap ``C_k`` samples its input
``V_k`` onto its bottom plate; releasing the top-plate switches lets
the caps redistribute charge until they reach a common node voltage.
Total charge conservation gives::

    V_out = sum_k C_k V_k / sum_k C_k

i.e. ``C_norm = C_total``.  The output is a passive **weighted
average** of the sampled voltages, not an unnormalised weighted sum.
Doubling every cap leaves ``V_out`` unchanged because both numerator
and denominator scale.  ``cap_ratios`` only set the *relative*
weights of the average — they are not directly equal to any
unnormalised "digit-weighted sum" semantics the upper layer might
want to attach.

Config vs. fabricate split
--------------------------
:class:`SwitchCapConfig` carries only the *physical* knobs — unit
capacitance, Pelgrom matching parameter, thermal-noise toggle,
energy / PPA accounting.  The bank's *virtual-instance shape* and
the *relative size* of caps inside one bank are supplied separately
to :meth:`SwitchCap.fabricate` as two arguments: ``shape`` (the
bank instance organisation) and ``cap_ratio`` (the 1-D per-cap
weight template).  This separation keeps the config purely
physical and matches the project-wide ``fabricate(shape, ...)``
contract — ``shape`` always describes the module's own virtual
instance organisation and never carries weight-template information.

Static mismatch / dynamic kT/C noise / energy
---------------------------------------------
* ``__init__`` registers a 0-d ``c_unit__fF`` buffer carrying the
  nominal unit cap.  ``c__fF`` (the fabricated per-cap cap tensor)
  is initialised to a fresh clone of the nominal and is overwritten
  on every :meth:`fabricate` call.
* :meth:`fabricate` rebuilds ``c__fF`` via
  ``c_unit__fF.clone().expand(shape).unsqueeze(-1) *
  cap_ratio.unsqueeze(0)`` and applies Pelgrom-scaled static
  mismatch.  The multiplication broadcasts the 0-d unit cap to the
  bank shape and the 1-D ``cap_ratio`` onto the trailing ``n_caps``
  axis.
* :meth:`sample_and_accumulate` is the compute kernel consumed by
  the readout's :meth:`forward`.  It separates two voltages:

  - ``v_in__V``: the driver-supplied input voltage that charges
    each cap (deterministic, given by the upstream stage).
  - ``v_hold__V``: the voltage held on the top plate **after** the
    sampling switch opens.  This is ``v_in__V`` plus the cap's
    kT/C settling fluctuation when ``enable_thermal_noise`` is on,
    or simply ``v_in__V`` when off.

  The held voltage drives the passive charge-share output; the
  *input* voltage drives the dynamic energy.  The split is
  physical: kT/C is a thermal settling fluctuation of the charge
  *already on* the cap once the switch isolates it, so it shows
  up only in the held-charge sum that resolves into ``V_out``.  It
  does **not** change the energy the driver had to pump in
  (``½ · C_k · V_in_k²``).
* The kT/C model uses :func:`apply_gaussian` with per-cap sigma::

      σ_V_k = sqrt(k_B · T / C_k)

  evaluated against the fabricated ``c__fF`` (so larger caps see
  smaller voltage noise — matching :class:`McsSarAdc`).  Unit
  conversion folds the project's standard ``fJ = fF · V²`` into a
  pre-computed ``kt__fJ = k_B · T · 1e15`` so the kernel can
  divide directly by ``c__fF`` (in fF) and get V² out.

  No per-VMM snapshot is threaded: the kT/C noise is drawn fresh
  inside this kernel on every call (the SAR-MCS convention),
  keeping the surface area minimal and matching the physical
  model where each sample-and-hold event is an independent
  thermal draw.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian, apply_pelgrom_mismatch
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K
from neurox.profiler import ProfiledModule


@dataclass(frozen=True, kw_only=True)
class SwitchCapConfig:
    """Immutable physical configuration for :class:`SwitchCap`.

    Holds physical knobs only — the bank's number and relative sizes
    of caps are supplied at :meth:`SwitchCap.fabricate` time via the
    ``ratio`` tensor, not via this config.  Operating temperature is
    also intentionally **not** here — it is an operating-state value
    passed to :class:`SwitchCap.__init__` as ``T__K`` (matching the
    convention used by :class:`~neurox.analog.adc.mcs_sar.McsSarAdc`).

    Attributes:
        c_unit__fF: Unit capacitance [fF].  Sets the absolute cap
            scale; the fabricated ``c__fF`` lives at this scale times
            the per-cap ``ratio``.  Must be ``> 0``.
        cap_mismatch_sigma_relative: Per-unit-cell Pelgrom relative
            sigma (eg. ``0.005`` for 0.5% matching), or ``None`` to
            disable.  Sigma per cap scales as ``sqrt(C_k / C_unit)``.
        enable_thermal_noise: Toggle for the per-cap kT/C settling
            noise added to the held voltage inside
            :meth:`SwitchCap.sample_and_accumulate`.  Affects the
            charge-share output but not the dynamic-energy term —
            kT/C is a thermal fluctuation of the *already-stored*
            charge, not extra driver work.  ``False`` yields a
            noise-free held voltage (``v_hold__V == v_in__V``).
        energy_per_sample_overhead__fJ: Constant per-bank switching
            overhead [fJ] added to the per-output energy.
        leakage_per_inst__uW: Static leakage per bank [uW].
        area_per_inst__um2: Silicon area per bank [um^2].
        latency_per_op__ns: Settling latency per sample [ns].
    """

    c_unit__fF: float

    cap_mismatch_sigma_relative: float | None

    enable_thermal_noise: bool

    energy_per_sample_overhead__fJ: float

    leakage_per_inst__uW: float
    area_per_inst__um2: float
    latency_per_op__ns: float

    def __post_init__(self) -> None:
        if not (self.c_unit__fF > 0.0):
            raise ValueError(f"SwitchCapConfig.c_unit__fF ({self.c_unit__fF}) must be > 0")
        if self.cap_mismatch_sigma_relative is not None and self.cap_mismatch_sigma_relative < 0.0:
            raise ValueError(
                f"SwitchCapConfig.cap_mismatch_sigma_relative ({self.cap_mismatch_sigma_relative}) must be >= 0 or None"
            )
        if self.energy_per_sample_overhead__fJ < 0.0:
            raise ValueError(
                f"SwitchCapConfig.energy_per_sample_overhead__fJ ({self.energy_per_sample_overhead__fJ}) must be >= 0"
            )
        if self.leakage_per_inst__uW < 0.0:
            raise ValueError(f"SwitchCapConfig.leakage_per_inst__uW ({self.leakage_per_inst__uW}) must be >= 0")
        if self.area_per_inst__um2 < 0.0:
            raise ValueError(f"SwitchCapConfig.area_per_inst__um2 ({self.area_per_inst__um2}) must be >= 0")
        if self.latency_per_op__ns < 0.0:
            raise ValueError(f"SwitchCapConfig.latency_per_op__ns ({self.latency_per_op__ns}) must be >= 0")


class SwitchCap(nn.Module, ProfiledModule):
    """Bottom-plate-sampled cap bank doing passive charge-share averaging.

    Args:
        cfg: Immutable :class:`SwitchCapConfig`.
        name: Hierarchical profiler name (e.g. ``<readout>.data_switchcap``).
        T__K: Operating temperature in Kelvin.  Drives the kT/C
            sampling-noise sigma evaluated inside
            :meth:`sample_and_accumulate`.  Per project convention
            temperature is an operating-state init arg, not a config
            (design) field.  Must be ``> 0``.
        dtype: Floating-point dtype for the fabricated ``c__fF``
            buffer and the in-kernel noise sample.

    Registered non-persistent buffers:
        ``c_unit__fF`` — 0-d scalar at
            :attr:`SwitchCapConfig.c_unit__fF`.  Built once at
            ``__init__``; never overwritten (the nominal template).
        ``c__fF`` — fabricated per-cap cap tensor.
            :meth:`fabricate(shape, cap_ratio)` rebuilds it on every
            call from ``c_unit__fF`` and the per-cap weight template
            ``cap_ratio``.
    """

    c_unit__fF: Tensor
    c__fF: Tensor

    def __init__(
        self,
        cfg: SwitchCapConfig,
        *,
        name: str = "",
        T__K: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        ProfiledModule.__init__(self, name)
        if not (T__K > 0.0):
            raise ValueError(f"SwitchCap.T__K ({T__K}) must be > 0")
        self.cfg = cfg
        self.T__K: float = T__K
        self.dtype = dtype

        # Nominal scalar — built once, never overwritten.
        self.register_buffer(
            "c_unit__fF",
            torch.tensor(cfg.c_unit__fF, dtype=dtype),
            persistent=False,
        )
        # Sentinel fabricated buffer — :meth:`fabricate` overwrites
        # it.  Initialised to a fresh clone of the nominal (no
        # expand) so ``.to(device)`` migrates cleanly even
        # pre-fabricate.
        self.register_buffer(
            "c__fF",
            self.c_unit__fF.clone(),
            persistent=False,
        )

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per bank [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per bank [uW]."""
        return self.cfg.leakage_per_inst__uW

    @property
    def latency_per_op__ns(self) -> float:
        """Settling latency per sample [ns]."""
        return self.cfg.latency_per_op__ns

    def fabricate(self, shape: tuple[int, ...], cap_ratio: Tensor) -> None:
        """Sample static cap mismatch from a bank shape + per-cap weight template.

        ``shape`` is the bank's virtual-instance organisation
        (typically ``(*prefix, group_num, data_num)`` from the
        readout); ``cap_ratio`` is the 1-D per-cap weight template
        (typically the offset code's
        ``(radix^0, radix^1, ..., radix^(D-1))``).  The fabricated
        cap tensor lands at ``(*shape, n_caps)``.

        Body:

        ``c__fF = c_unit__fF.clone().expand(shape).unsqueeze(-1)
                  * cap_ratio.unsqueeze(0)``

        broadcasts the 0-d unit cap up to the bank shape and the
        1-D ``cap_ratio`` onto the trailing ``n_caps`` axis.
        Pelgrom-scaled static mismatch is then applied on top.

        Args:
            shape: Bank instance organisation, ``(*prefix, ...)``.
                The fabricated ``c__fF`` ends up at
                ``(*shape, n_caps)``.
            cap_ratio: 1-D per-cap weight template of shape
                ``[n_caps]``.  Cast to ``self.dtype`` internally
                before broadcasting onto the n_caps axis.
        """
        cfg = self.cfg
        weights = cap_ratio.to(self.dtype)
        c__fF = self.c_unit__fF.clone().expand(shape).unsqueeze(-1) * weights.unsqueeze(0)
        c__fF = apply_pelgrom_mismatch(
            c__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )
        self.register_buffer("c__fF", c__fF, persistent=False)

    def sample_and_accumulate(self, v_in__V: Tensor) -> Tensor:
        """Sample digit voltages and run passive charge-sharing.

        Two voltages flow through this kernel:

        - ``v_in__V``: driver-supplied input voltage on the bottom
          plate.  Deterministic — set by the upstream stage.
        - ``v_hold__V``: top-plate voltage held after the sampling
          switch opens.  Equals ``v_in__V`` plus a per-cap kT/C
          settling fluctuation when ``enable_thermal_noise`` is on
          (the dynamic-noise model — see module docstring), or
          ``v_in__V`` unchanged when off.

        Dynamic energy is emitted through the profiler side channel
        rather than returned to the caller.

        Args:
            v_in__V: Per-cap sampled (driver-supplied) voltages [V].
                Shape ``(*batch, *bank_shape, n_caps)``.

        Returns:
            Node voltage with shape ``(*batch, *bank_shape)``.
        """
        c__fF = self.c__fF
        if self.cfg.enable_thermal_noise:
            # kT/C settling noise on the held top-plate voltage.
            # ``kt__fJ = k_B · T · 1e15`` folds the fJ = fF · V² unit
            # convention so ``kt__fJ / c__fF`` lands in V² directly.
            kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
            sigma__V = torch.sqrt(kt__fJ / c__fF)
            v_hold__V = apply_gaussian(v_in__V, sigma__V)
        else:
            v_hold__V = v_in__V
        # Passive charge share: V_out = Σ Q_k / Σ C_k, with Q_k taken
        # from the *held* voltage (the kT/C noise enters the output).
        c_total__fF = c__fF.sum(dim=-1)
        v_out__V = torch.sum(c__fF * v_hold__V, dim=-1) / c_total__fF

        # Driver-side dynamic energy emitted as a profiler event.
        e_caps__fJ = 0.5 * torch.sum(c__fF * v_in__V * v_in__V, dim=-1)
        dynamic_energy__fJ = e_caps__fJ + self.cfg.energy_per_sample_overhead__fJ
        self._log_dynamic(dynamic_energy__fJ, self.cfg.latency_per_op__ns)
        return v_out__V

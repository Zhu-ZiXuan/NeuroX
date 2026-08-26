"""Run configuration for ADC-input characterization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from neurox.common import ConfigBase, ValidateMixin
from neurox.tools._macro import MacroSection


class ActiveRowSelection(StrEnum):
    SCATTERED = "scattered"
    CONTIGUOUS = "contiguous"


@dataclass(frozen=True)
class ProbeConfig(ValidateMixin):
    quantization_mode: int

    def __post_init__(self) -> None:
        self._require_non_neg(self.quantization_mode, "[probe].quantization_mode")


@dataclass(frozen=True)
class RandomStimulusConfig(ValidateMixin):
    seed: int
    weight_samples: int
    input_samples_per_weight: int
    batch_w: int
    batch_x: int

    def __post_init__(self) -> None:
        self._require_non_neg(self.seed, "[stimulus.random].seed")
        self._require_pos(self.weight_samples, "[stimulus.random].weight_samples")
        self._require_pos(self.input_samples_per_weight, "[stimulus.random].input_samples_per_weight")
        self._require_pos(self.batch_w, "[stimulus.random].batch_w")
        self._require_pos(self.batch_x, "[stimulus.random].batch_x")
        if self.weight_samples % self.batch_w:
            raise ValueError("require: weight_samples is divisible by batch_w")


@dataclass(frozen=True)
class TargetStimulusConfig(ValidateMixin):
    seed: int
    batch_w: int

    def __post_init__(self) -> None:
        self._require_non_neg(self.seed, "[stimulus.target].seed")
        self._require_pos(self.batch_w, "[stimulus.target].batch_w")


@dataclass(frozen=True)
class StimulusConfig:
    active_row_selection: ActiveRowSelection
    random: RandomStimulusConfig
    target: TargetStimulusConfig


class AdcProbeToolConfig(ConfigBase):
    macro: MacroSection
    probe: ProbeConfig
    stimulus: StimulusConfig

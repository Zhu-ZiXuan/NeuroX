"""Generic macro-ADC calibration tools (config-driven, scheme-agnostic).

Three CLI entries built on the :class:`~neurox.common.prober.AdcProber`
calibration channels and the ``CimMacro`` registry:

- :mod:`.rescale_fit` — dual physical/ideal probed run; zero-through-origin
  LS fit of the per-(mode, bits) ``rescale_factor``; ``[[adc_calibration]]``
  fragment.
- :mod:`.threshold_probe` — controlled-stimulus ``I(M)`` grid sweep;
  mid-point threshold placement with band-margin analysis; ladder fragment.
- :mod:`.mode_derive` — mode-set derivation from a per-layer range mapping
  file (sign-group split, 1-D clustering); emits the mode-set TOML the two
  calibration CLIs consume.

The two TOML file formats the tools exchange (the layer-range mapping and
the mode set) are owned by :mod:`._modes`.

See also:
    docs/guides/calibration/calibrate_adc.md
"""

# Analog Modules

This directory documents the current analog/circuit layer.

Analog configs follow the circuit-layer rule:

- configs contain design + spec parameters
- configs also contain member configs for owned child devices / circuits
- runtime parameters stay out of config

Owned children are constructed directly by the parent class; external factory closures are not part of the current design.

## Leaf circuits

These are not polymorphic families — each is a single concrete `nn.Module` the consuming circuit constructs directly:

- [`analog_mux.md`](analog_mux.md) — differential transport with optional CM / DM noise.
- [`clamp_driver.md`](clamp_driver.md) — the shared solver-facing `ClampDriver` protocol (TIA and `Driver` both implement it).
- [`decoder.md`](decoder.md) — row decoder + WL driver wrapper.
- [`driver.md`](driver.md) — ideal constant-voltage clamp driver.
- [`switch_cap.md`](switch_cap.md) — bottom-plate-sampled cap bank.

## Polymorphic families

Each sub-directory holds a polymorphic family driven by `RegistryMixin` keyed on the concrete config class:

- [`dac/`](dac/README.md) — `DAC` family (today: `GeneralDAC`).
- [`adc/`](adc/README.md) — `ADC` family (today: `GeneralADC`, `McsSarAdc`, `SarAdcMono`).
- [`tia/`](tia/README.md) — `TIA` family (today: `OpAmpTIA`).

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/adr/ADR-0001-config-dispatch-and-owned-construction.md`

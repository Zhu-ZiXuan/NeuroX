# ADC Calibration — Two-Fold Output (Range + Precision)

`neurox.tools.xbar_adc_boundaries` calibrates the chip's
highest-precision ADC mode by sweeping `(weight, activation)` pairs
through the real physical xbar and a corresponding ideal xbar, then
fitting floor-style boundaries.

## CLI

```
python -m neurox.tools.xbar_adc_boundaries \
    --config path/to/chip.toml \
    --random N           # N random input groups; <= 0 traverses corner cases
    --noise              # apply physical noise during sampling
    --visualize          # save histogram PNG alongside --output
    --output path.toml   # write calibrated [bl_adc] block
```

When `--output` is omitted the calibrated TOML block prints to
stdout.

## Algorithm

1. Build the **physical xbar** and a corresponding **ideal xbar**
   from the supplied chip TOML.  `--noise` toggles whether physical
   noise sub-configs (programming Gamma, read thermal, etc.) are
   active.
2. Generate input groups:
   - `--random N` (default 256): sample `N` random
     `(weight, activation)` matrices uniformly across legal integer
     states.
   - `N <= 0`: enumerate the four canonical corner cases (HRS / LRS
     extremes, checkerboard, all-zero).
3. For each group:
   - Run the physical xbar's analog path up to the ADC, capturing
     the **pre-quantization BL signal** `s` (mA, post-TIA voltage if
     a TIA is configured).
   - Run the ideal xbar to compute the **exact ideal integer code**
     `c_ideal`.
   - Pair `(s, c_ideal)` into the dataset.
4. **Two-fold output:**
   - **Range**: empirical `s_max = max(|s|)` defines `max_signal`
     for the highest-precision mode.
   - **Precision**: `n_bits_max = ceil(log2(max(c_ideal) + 1))`
     defines `n_bits` and `n_states` for the highest-precision mode.
5. **Boundaries** for the highest-precision mode placed at
   `c · LSB` (floor semantics) where `LSB = max_signal / n_codes`.
6. **Visualization** (`--visualize`): histogram of `s` with
   vertical bars at the chosen boundaries; saved as PNG.

## Floor semantics

The current code uses `floor_boundaries_for_mode` which produces
`c · LSB` boundaries (NOT the legacy `(c - 0.5) · LSB` round-to-nearest
form).  This matches the new ADC family's `floor_bucketize` semantics
and the `FakeXbar._dot_to_code` floor computation — eliminating the
historical mismatch between the fake reference and the physical
ADC.

## Lower-precision modes

Lower-precision modes are derived from the highest by halving codes
and shrinking `max_signal` proportionally:

```python
modes = derive_modes(
    n_bits_max=8, s_max=1.2, n_states_max=193,
    additional_modes=[(6, 64), (4, 128)],
)
```

The macro's `output_rescale_factor` automatically picks up the
selected mode's `n_states`.

## Train-time minimum-range refinement

A future iteration will add an EMA observer on each xbar's
pre-quantization signal during HAT Phase 1, then at HAT freeze pick
the smallest preconfigured mode whose `max_signal` covers the
observed value.  For now, calibration runs offline as a one-shot CLI
step before training.

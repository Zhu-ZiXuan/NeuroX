# For Algorithm Engineers

Goal: evaluate how your model's accuracy and energy behave when its linear layers run on a real RRAM crossbar instead of ideal arithmetic.

The supported workflow is **manual operator replacement**: define a quantized model whose conv / linear layers are backed by `neurox.macro`, train (or load) a QAT checkpoint, and evaluate against either the lossless `IdealXbarMacro` or a physical-xbar chip configured by `--config` / `--policy` TOML files. The public surface stops at `neurox.macro`; everything under `example/` is user code, not part of the API.

Two bundled end-to-end examples demonstrate the full flow:

- **LeNet-5 on MNIST** (`example/lenet/`) — small CNN, fast to train; exercises the macro / xbar / readout / ADC chain on a workload whose activation range lands easily on the chip's quantization grid.
- **BERT-small on SST-2** (`example/bert/`) — transformer encoder fine-tuning; exercises the same chain on a much wider activation range and deeper layer count.

## What each example contains

For each model directory:

- `model_float.py` / `model_quant.py` — float reference and QAT / macro-quantized model definitions.
- `data.py` — dataset loader (MNIST or SST-2).
- `train_float.py` / `train_quant.py` — float pretraining and QAT scripts. QAT takes the float checkpoint and emits a QAT checkpoint; it never touches the chip.
- `evaluate.py` — evaluation against either the lossless `IdealXbarMacro` or the physical-xbar chip.
- `macro_factory.py` — loads the circuit config (`--config`) and the nonideality policy (`--policy`) from their TOML files and builds one macro per layer.
- `quant.py` — per-layer quantized conv / linear operators.

## How to run

The bundled `make` targets wire the right checkpoint, config, and policy for each model:

```bash
make eval-lenet MAX_SAMPLES=100 BATCH_SIZE=10    # physical xbar, MNIST
make eval-bert  MAX_SAMPLES=100                  # physical xbar, SST-2
```

The equivalent direct invocation spells out the two config files — the immutable circuit `--config` and the mutable nonideality `--policy`:

```bash
python -m example.lenet.train_float --device cuda:0 --checkpoint weight/lenet_float.pth
python -m example.lenet.train_quant --device cuda:0 \
    --float-checkpoint weight/lenet_float.pth --checkpoint weight/lenet_qat.pth
python -m example.lenet.evaluate    --device cuda:0 --dataset-dir dataset/mnist \
    --checkpoint weight/lenet_qat.pth \
    --config macro_with_physical_xbar.toml --policy macro_with_physical_xbar.policy.toml
```

Swap `--config macro_with_ideal_xbar.toml --policy macro_with_ideal_xbar.policy.toml --xbar ideal` for the lossless baseline. The physical path is memory-heavy (im2col explodes the solver's leading batch); keep `--batch-size` small first, then set the chunking knob (`solve_chunk_size`) in the policy file (see below).

The examples default to GPU because both workloads are too slow on CPU to be useful as training references. Pass `--device cpu` explicitly when GPU is unavailable (LeNet is feasible on CPU; BERT is not).

## Circuit config and nonideality policy

Each macro is built from two separate TOML files; the full schema and the `_neurox_*` directives are in [API: configuration](../../api/configuration.md).

- **`--config`** — the immutable circuit design. Both examples consume a bundled 1T1R 28nm preset, shipped with its scheme and referenced from `macro_with_physical_xbar.toml` via `_neurox_use`. To target a different chip, follow the same `[xbar]` / `[xbar.core_config]` schema (the driver and readout blocks are inline `[xbar.*]` sections above the core). The ADC `rescale_factor` table must be calibrated for the chip's `(adc_mode, adc_bits)` grid — see the [calibration guide](../calibration/README.md).
- **`--policy`** — the mutable nonideality switches. The example policy files reference the bundled all-off preset (`neurox/presets/policy/all_off.toml`) via `_neurox_use_preset`, so every nonideality (device mismatch, thermal noise, programming noise, ADC offsets, ...) is off by default. To enable one, override the matching switch inline:

  ```toml
  [policy]
  _neurox_use_preset = "policy/all_off:macro"

  [policy.xbar.core.tia]
  opamp_gain_sigma = true
  ```

  The solver chunking knob (`solve_chunk_size`) also lives here, under `[policy.xbar.core]`.

## See also

- [API: configuration](../../api/configuration.md) — the `--config` / `--policy` TOML schema and presets.
- [Reference: macro](../../reference/macro/README.md) — the `XbarMacro` family and `from_config` factory.
- [Reference: xbar solver](../../reference/xbar/solver.md) — the topology-agnostic physical-array solve: the single nested formulation over the pluggable cell and clamp drivers.
- [Calibration guide](../calibration/README.md) — calibrating the ADC `rescale_factor` table for a new chip.

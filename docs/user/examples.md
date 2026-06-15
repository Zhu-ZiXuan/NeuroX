# Example Pipelines

NeuroX ships two end-to-end examples under `example/` that wire the core library into a runnable training + inference flow:

- **LeNet-5 on MNIST** (`example/lenet/`) — small CNN, fast to train; exercises the macro / xbar / readout / ADC chain on a workload where the chip's quantisation grid lands easily.
- **BERT-small on SST-2** (`example/bert/`) — transformer encoder fine-tuning; exercises the same chain on a much wider activation range and deeper layer count.

Both examples are *not* part of the public NeuroX API. They are reference assemblies showing how a user wires `neurox.macro` into a training loop, plugs in QAT observers, and runs `analyze_model` / `analyze_static` against a chip TOML. The core public surface stops at `neurox.macro` — anything in `example/` is user code.

## What each example contains

For each model directory you will find:

- `model_float.py` / `model_quant.py` — float reference and QAT/macro-quantised model definitions.
- `data.py` — dataset loader (MNIST or SST-2).
- `train_float.py` / `train_quant.py` — float pretraining and QAT scripts (QAT takes the float checkpoint and emits a QAT checkpoint; it never touches the chip).
- `evaluate.py` — evaluation against either the lossless `IdealXbarMacro` or the physical `Offset1T1RXbar` chip.
- `macro_factory.py` — loads the circuit config (`--config`) and the nonideality policy (`--policy`) from their TOML files and builds one macro per layer.
- `quant.py` — per-layer quantised conv / linear operators.

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

Swap `--config macro_with_ideal_xbar.toml --policy macro_with_ideal_xbar.policy.toml --xbar ideal` for the lossless baseline. The physical path is memory-heavy (im2col explodes the solver's leading batch); keep `--batch-size` small or set the chunking knobs in the policy file (see below).

The examples default to GPU because both workloads are too slow on CPU to be useful as training references. Pass `--device cpu` explicitly when GPU is unavailable (LeNet is feasible on CPU; BERT is not).

## Circuit config and nonideality policy

Each macro is built from two separate TOML files:

- **`--config`** — the immutable circuit design. Both examples consume the bundled 1T1R 28nm preset at `example/config/1t1r_28nm.toml` (referenced from `macro_with_physical_xbar.toml` via `_neurox_use`). To target a different chip, follow the same `[xbar]` / `[xbar.core_config]` / `[xbar.readout_config]` schema. The ADC `rescale_factor` table must be calibrated for the chip's `(adc_mode, adc_bits)` grid — see `neurox.tools.xbar_adc.calibrate` and `docs/modules/tools/xbar_adc/calibrate.md`.
- **`--policy`** — the mutable nonideality switches. The example policy files reference the bundled all-off preset (`neurox/presets/policy/all_off.toml`) via `_neurox_use_preset`, so every nonideality (device mismatch, thermal noise, programming noise, ADC offsets, …) is off by default. To enable one, override the matching switch inline, e.g.:

  ```toml
  [policy]
  _neurox_use_preset = "policy/all_off:macro"

  [policy.xbar.core.tia]
  opamp_gain_sigma = true
  ```

  The solver chunking knobs (`solve_chunk_size_x` / `solve_chunk_size_inst`) also live here, under `[policy.xbar.core]`.

## Related developer docs

- [`docs/dev/architecture/README.md`](../dev/architecture/README.md) — project architecture rules.
- [`docs/modules/macro/README.md`](../modules/macro/README.md) — `XbarMacro` family and `from_config` factory.
- [`docs/modules/xbar/_1t1r/solver.md`](../modules/xbar/_1t1r/solver.md) — solver family base. [`nested_solver.md`](../modules/xbar/_1t1r/nested_solver.md) is the production default; [`full_jacobian_solver.md`](../modules/xbar/_1t1r/full_jacobian_solver.md) is the debug / cross-check reference.

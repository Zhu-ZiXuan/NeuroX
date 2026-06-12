# Example Pipelines

NeuroX ships two end-to-end examples under `example/` that wire the core library into a runnable training + inference flow:

- **LeNet-5 on MNIST** (`example/lenet/`) — small CNN, fast to train; exercises the macro / xbar / readout / ADC chain on a workload where the chip's quantisation grid lands easily.
- **BERT-small on SST-2** (`example/bert/`) — transformer encoder fine-tuning; exercises the same chain on a much wider activation range and deeper layer count.

Both examples are *not* part of the public NeuroX API. They are reference assemblies showing how a user wires `neurox.macro` into a training loop, plugs in QAT observers, and runs `analyze_model` / `analyze_static` against a chip TOML. The core public surface stops at `neurox.macro` — anything in `example/` is user code.

## What each example contains

For each model directory you will find:

- `model_float.py` / `model_quant.py` — float reference and QAT/macro-quantised model definitions.
- `data.py` — dataset loader (MNIST or SST-2).
- `train_float.py` / `train_quant.py` — float training and HAT scripts.
- `evaluate.py` — evaluation against either the lossless `IdealXbarMacro` or the physical `Offset1T1RXbar` chip.
- `macro_factory.py` — chip-config-to-macro factory plus an all-off policy helper.
- `quant.py` — per-layer quantised conv / linear operators.

## How to run

LeNet:

```bash
python -m example.lenet.train_float --device cuda:0
python -m example.lenet.train_quant --device cuda:0 --chip-config example/config/1t1r_28nm.toml
python -m example.lenet.evaluate    --device cuda:0 --chip-config example/config/1t1r_28nm.toml
```

BERT:

```bash
python -m example.bert.train_float  --device cuda:0
python -m example.bert.train_quant  --device cuda:0 --chip-config example/config/1t1r_28nm.toml
python -m example.bert.evaluate     --device cuda:0 --chip-config example/config/1t1r_28nm.toml
```

The examples default to GPU because both workloads are too slow on CPU to be useful as training references. Pass `--device cpu` explicitly when GPU is unavailable (LeNet is feasible on CPU; BERT is not).

## Chip configuration

Both examples consume the bundled 1T1R 28nm preset at `example/config/1t1r_28nm.toml`. To target a different chip, point `--chip-config` at your own TOML following the same `[xbar]` / `[xbar.core_config]` / `[xbar.readout_config]` schema. The ADC `rescale_factor` table in that file must be calibrated for the chip's `(adc_mode, adc_bits)` grid — see `neurox.tools.xbar_adc.calibrate` and `docs/dev/modules/tools/xbar_adc/calibrate.md`.

## Related developer docs

- [`docs/dev/architecture/README.md`](../dev/architecture/README.md) — project architecture rules.
- [`docs/dev/modules/macro/README.md`](../dev/modules/macro/README.md) — `XbarMacro` family and `from_config` factory.
- [`docs/dev/modules/xbar/_1t1r/solver.md`](../dev/modules/xbar/_1t1r/solver.md) — solver family base. [`nested_solver.md`](../dev/modules/xbar/_1t1r/nested_solver.md) is the production default; [`full_jacobian_solver.md`](../dev/modules/xbar/_1t1r/full_jacobian_solver.md) is the debug / cross-check reference.

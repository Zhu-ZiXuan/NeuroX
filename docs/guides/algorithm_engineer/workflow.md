# Algorithm-engineer workflow

Goal: measure how a model's accuracy and energy behave when its conv and linear layers run on a simulated RRAM crossbar instead of ideal arithmetic.

The route is manual operator replacement. Define a quantized model whose conv and linear layers are backed by `neurox.architecture.unit`, train it to a QAT checkpoint, then evaluate that checkpoint against a chip described by a `--config` / `--policy` file pair. The public surface stops at `neurox.architecture.unit` ([Python API](../../api/python.md)); everything under `example/` is user code demonstrating the route, not part of it.

The bundled pipelines walk the whole route end to end:

- **LeNet-5 on MNIST** (`example/lenet/`) — a small CNN, fast to train, whose activation range lands easily on the chip's quantization grid.
- **BERT-small on SST-2** (`example/bert/`) — a transformer encoder over a much wider activation range and a deeper layer count. It needs `transformers` and `datasets` beyond the core dependencies.

Both directories carry the same set of scripts: `model_float.py` and `model_quant.py` (the float model and its quantized twin), `data.py` (the dataset loader), `train_float.py` and `train_quant.py` (the two training stages), `quant.py` (the per-layer quantized conv and linear operators), `macro_factory.py` (config and policy loading, one unit per layer), and `evaluate.py` (the measured run).

Every runnable script takes `--device` and defaults to a CUDA device, so pass `--device cpu` explicitly when no GPU is available. LeNet is feasible on CPU; BERT is not.

## Step 1 — train the float reference

Float training touches no simulated hardware. It produces the teacher and the initialization the next stage starts from.

```bash
python -m example.lenet.train_float --device cuda:0 --dataset-dir dataset/mnist \
    --checkpoint weight/lenet_float.pth
```

The BERT pipeline fine-tunes instead of training from scratch, with the same three flags plus `--max-length`:

```bash
python -m example.bert.train_float --device cuda:0 --dataset-dir dataset/sst2 \
    --checkpoint weight/bert_small_float.pth
```

## Step 2 — quantization-aware training

QAT consumes the float checkpoint and emits a QAT checkpoint. No macro runs in its forward pass: it settles the per-layer observers and trains the model on the quantization grid the chip will present, distilling from the frozen float teacher.

```bash
python -m example.lenet.train_quant --device cuda:0 --dataset-dir dataset/mnist \
    --float-checkpoint weight/lenet_float.pth --checkpoint weight/lenet_qat.pth
```

`--calibration-batches` sets how many forward-only train-mode batches run before training so the observers settle; `--kd-alpha` and `--kd-temperature` weigh the hard-label and distillation terms.

## Step 3 — evaluate against the chip

Evaluation is where the crossbar enters. `--config` and `--policy` name TOML files inside the pipeline directory, not paths, and the checkpoint must be the one step 2 wrote — the script rejects a checkpoint whose schema tag does not match.

```bash
python -m example.lenet.evaluate    --device cuda:0 --dataset-dir dataset/mnist \
    --checkpoint weight/lenet_qat.pth \
    --config macro_with_physical_xbar.toml --policy macro_with_physical_xbar.policy.toml
```

`--max-samples` caps the samples processed, which is the knob for a quick smoke run.

The bundled `make` targets bind the checkpoint, config, policy, and batch size per model, on a shared device, and each is overridable on the command line (`DEVICE=`, `BATCH_SIZE=`, `EVAL_CKPT=`, `CONFIG=`, `POLICY=`, `CIM_MACRO=`):

```bash
make eval-lenet MAX_SAMPLES=100 BATCH_SIZE=10
make eval-bert  MAX_SAMPLES=100
```

## Step 4 — read the report

The run names the config and policy it used, then prints accuracy beside the PPA figures: total area, total leakage power, total dynamic energy, and the modeled latency per sample, followed by the per-module dynamic-energy breakdown that shows where the energy went.

Leakage power and latency stay separate figures rather than being multiplied into a static energy: static energy is leakage times the duty-cycle period a deployment holds the macro for, which is a property of that deployment and not of the access time. The axes behind these numbers are in [PPA accounting](../../system_design/ppa_accounting.md).

The measurement objects are public, so the same readout works in your own evaluation script: `neurox.stamp_names` names the assembled model once, `neurox.Profiler` collects the records one measured call emits, and `neurox.Reporter` turns the model plus that profiler into the static and dynamic rows ([Common API](../../api/common.md)).

## Comparing against a lossless reference

`--cim_macro ideal` swaps the configured tile for its `to_ideal()` twin — the faithful lossless reference of that same chip, sharing its geometry and value domains — while `--cim_macro physical` runs the tile as configured. Running the same command both ways isolates the analog loss from the QAT loss.

A standalone ideal config is the other route: `macro_with_ideal_xbar.toml` for LeNet and `macro_ideal.toml` for BERT, each with its own policy file. These instantiate an ideal tile directly from hand-authored parameters tied to no fabricated chip, so they serve flow bring-up and carry no hardware provenance.

## The two config files

A run is described by two files, whose schema, `_neurox_*` directives, and preset mechanism are specified in [Configuration](../../api/configuration.md):

- **`--config`** — the immutable circuit design. In the bundled files the top section is `[cim_unit]`, and the tile it drives sits at `[cim_unit.engine.cim_macro_config]`, either tagged with `_neurox_class` or pulled from a scheme's chip params by `_neurox_use`. To target a different chip, point that section at that chip's params.
- **`--policy`** — the mutable nonideality switches, mirroring the config's section tree. The example policies pull in a scheme's all-off preset, so every nonideality starts off; enable one by overriding its `bool` inline after the `_neurox_use` line that pulls the preset in.

Each quantization mode the chip declares carries its own calibrated rescale factor, derived by the [calibration guides](../calibration/README.md). That factor states an output code in ideal-macro codes; turning codes into MAC units is the model's own job and multiplies in the mode's window step, which is what `quant.py` folds into its per-channel scales. Train and evaluate against the same config pair, or the folded scales no longer match the codes the chip returns.

## Bounding memory on the physical path

The physical path is memory-heavy: a conv layer's window expansion puts one row per output position on the leading batch the array solve sees. Keep `--batch-size` small on the first run, then tune `solve_chunk_size` under the array's policy table. The knob bounds the peak memory of one call and moves no number, so a result is bit-identical across chunk sizes; it has to be tuned against the compiled path, whose per-chunk peak exceeds an eager run's at the same size. Both properties are stated in [crossbar DC solve](../../system_design/xbar_solve.md) and [compile boundary](../../system_design/compile.md).

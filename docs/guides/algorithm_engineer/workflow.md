# Algorithm-engineer workflow

Train an original floating-point model, then observe its operator energy with NeuroX during checkpoint inference. The original model supplies the predictions; energy observation checks that each sample's logits match inference without observation exactly.

The bundled examples provide LeNet-5 on MNIST and BERT-small on SST-2. Each directory contains `model_float.py`, `data.py`, `train_float.py`, and an `evaluate.py` entry into the shared energy evaluation code. BERT requires `transformers` and `datasets` in addition to the core dependencies.

## Train the original model

LeNet trains from random initialization and saves a floating-point state dictionary:

```bash
python -m example.lenet.train_float --device cuda:0 --dataset-dir dataset/mnist \
    --checkpoint weight/lenet_float.pth
```

BERT fine-tunes the pretrained encoder and classification head:

```bash
python -m example.bert.train_float --device cuda:0 --dataset-dir dataset/sst2 \
    --checkpoint weight/bert_small_float.pth
```

The corresponding Make targets are `make train-lenet DEVICE=cuda:0` and `make train-bert DEVICE=cuda:0`. Training parameters can be overridden through `DATASET_DIR`, `RAW_CKPT`, `BATCH_SIZE`, `EPOCHS`, and `LR`; BERT also accepts `MAX_LENGTH`.

## Evaluate energy

Evaluation loads the original checkpoint and observes supported linear and convolution operators:

```bash
python -m example.lenet.evaluate --device cuda:0 --checkpoint weight/lenet_float.pth \
    --preset xue2020jssc --num-samples 10 --output log/energy/lenet_xue.json
python -m example.bert.evaluate --device cuda:0 --checkpoint weight/bert_small_float.pth \
    --preset ye2023jssc --num-samples 10 --output log/energy/bert_ye.json
```

Checkpoints and evaluation data must be available locally. `--num-samples` bounds the evaluation size. `--batch-chunk` and `--spatial-chunk` bound the work submitted to each measurement call. The Make targets `eval-lenet` and `eval-bert` accept `DEVICE`, `EVAL_CKPT`, `PRESET`, and `MAX_SAMPLES`.

The energy observer maps operands using the bundled circuit presets and runs ideal macro twins. It reports configured digital-operation energy; analog dynamic energy is excluded. Operand encoding belongs to the independent measurement and does not alter the original model. Attention matrix products and operations outside the observed linear/convolution products are outside this report's scope.

## Read the report

The JSON output records the checkpoint, preset, sample count, correct predictions, whether logits were preserved, and the operator energy breakdown. Digital accumulator and shift-adder costs are explicit evaluation assumptions, so these results establish integration rather than calibrated physical-macro energy estimates.

For custom measurement code, `neurox.stamp_names` names an assembled module tree, `neurox.Profiler` collects dynamic-energy records, and `neurox.Reporter` aggregates static metrics and dynamic energy ([Python API](../../api/python.md)). Static metrics include leakage power; converting it to static energy requires a measurement period chosen by the caller, as described in [PPA accounting](../../system_design/ppa_accounting.md).

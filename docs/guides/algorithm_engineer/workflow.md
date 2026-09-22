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

Checkpoints and evaluation data must be available locally. `--num-samples` bounds the evaluation size; each selected sample forms one inference batch. The Make targets `eval-lenet` and `eval-bert` accept `DEVICE`, `EVAL_CKPT`, `PRESET`, and `MAX_SAMPLES`.

The observer uses physical macro presets by default; `--ideal-macro` selects ideal twins and excludes analog dynamic energy. Each logical operator position owns one unit programmed before measurement. Operand encoding preserves the original model's outputs. Grouped convolutions use independent hardware per group. Attention matrix products and operations outside the observed linear/convolution products are outside the report's scope. Unsupported input domains are reported in `<output>.unsupported.json` and stop preparation.

## Read the report

The JSON output records the checkpoint, preset, sample count, accuracy, logit-preservation check, and per-sample and per-operator PPA. Digital circuit costs and clock timing are evaluation assumptions defined in `example/energy/factory.py`, not measurements from the cited papers.

The accompanying `.profile.pt` file retains named observations for custom analysis through the [Python API](../../api/python.md). Static energy defaults to each unit's working window; alternative supply-on schedules use powered-window overrides under [PPA accounting](../../system_design/ppa_accounting.md). Application analysis groups hardware and reduces operation axes to dataset samples.

# BERT-small on SST-2

End-to-end NeuroX example: fine-tune BERT-small on SST-2, hardware-aware
QAT through a crossbar macro, then evaluate the NeuroX-flat checkpoint
against any macro backend (`fake`, `xbar_ideal`, `xbar1t1r`).

## Model + dataset

- **Model**: `prajjwal1/bert-small` (4 layers, 512 hidden, 8 heads, ~28M params)
  with a 2-class sequence-classification head.
- **Dataset**: GLUE / SST-2 binary sentiment (loaded via HuggingFace
  `datasets`).
- **Tokenizer**: HuggingFace `AutoTokenizer`, padded / truncated to
  `--max-length` (default 128).

NeuroX `replace_for_hat` and `build_evaluator` walk the module tree and
swap **every `nn.Linear`** — Q/K/V/output attention projections, FFN
intermediate / output, the `[CLS]` pooler, and the classifier — for the
crossbar-backed counterpart.  Embeddings, LayerNorm, GELU, and the
attention softmax stay in float.

## Extra dependencies

This example needs `transformers` and `datasets` on top of NeuroX's
core deps:

```bash
pip install transformers datasets
```

## Pipeline

```bash
# 1. Float fine-tune (≈3 epochs, ~5 min on a single GPU)
python -m example.bert.train \
  --dataset-dir dataset/sst2 \
  --checkpoint weight/bert_small_float.pth \
  --device cuda:1 \
  --epochs 3

# 2. Hardware-aware QAT with knowledge distillation through the ideal-xbar macro
python -m example.bert.hat_qat \
  --dataset-dir dataset/sst2 \
  --float-checkpoint weight/bert_small_float.pth \
  --checkpoint weight/bert_small_hat.pth \
  --device cuda:0 \
  --epochs 3 \
  --kd-alpha 0.3 --kd-temperature 2.0 --kd-hidden-weight 1.0

# 3. Evaluate the NeuroX-flat checkpoint
python -m example.bert.evaluate \
  --dataset-dir dataset/sst2 \
  --checkpoint weight/bert_small_hat.pth \
  --device cuda:0 \
  --xbar physical
```

## Notes

- **Quantization grid**: weight / activation grids match
  `example/bert/macro.toml` (4-state RRAM × 1-digit weight ⇒ ±3,
  2-state × 4-digit activation ⇒ [0, 15]).  The output grid defaults
  to signed 8-bit (`y_qmin=-128`, `y_qmax=127`) so BERT's post-layer
  activation range survives requantization.
- **Hardware is TOML-driven**: every circuit parameter — ADC
  resolution (`[bl_adc].boundaries`), RRAM states, switch, wires,
  digital datapath, tile geometry, rescale factor — is read from
  `example/bert/macro.toml` (which pulls the 1T1R xbar reference from
  `example/config/1t1r_28nm.toml` via `_neurox_use`). To
  change the ADC resolution, edit the `[bl_adc].boundaries` list and
  add a matching `[[xbar.adc_calibration]]` record accordingly; the CLI
  has no hardware knobs.
- **ADC resolution and depth**: the bundled 16-level (4-bit) ADC
  collapses BERT-small's signal to chance accuracy — 26 linear
  layers compound per-tile quantization noise past the signal floor.
  For deep transformer stacks, configure a finer ADC (e.g. 193-level
  lossless) in the TOML before running HAT / eval.
- **Knowledge distillation**: a frozen float teacher supervises the
  HAT student via soft-label KL (temperature `T=2.0`) plus pooled
  `[CLS]` MSE.  Combined with a high-resolution ADC, one epoch lands
  at ≈ 83 % val_acc — comfortably above the 80 % bar.  Without KD
  the same student stalls around chance.
- **HAT cost**: BERT-small has 4 encoder blocks × 6 linears each + pooler
  + classifier = 26 linear layers, each running through the macro on
  every forward.  HAT training is much slower than float fine-tuning;
  start with small batch size (16) and few epochs (1-3).
- **Sequence length**: SST-2 sentences are short (median ≈ 11 tokens);
  `--max-length 128` is comfortable but you can drop to 64 for speed.
- **TOML consistency between HAT and eval**: train and evaluate with
  the same `example/bert/macro.toml` so the saved
  `(multiplier, rshift, bias_int)` buffers align with the evaluator's
  rescale factor.

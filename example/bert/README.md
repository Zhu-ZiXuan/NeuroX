# BERT-small on SST-2

End-to-end NeuroX example: fine-tune BERT-small on SST-2, hardware-aware QAT through a crossbar macro, then evaluate the NeuroX-flat checkpoint against any macro backend (`fake`, `xbar_ideal`, `xbar1t1r`).

## Model + dataset

- **Model**: `prajjwal1/bert-small` (4 layers, 512 hidden, 8 heads, ~28M params) with a 2-class sequence-classification head.
- **Dataset**: GLUE / SST-2 binary sentiment (loaded via HuggingFace `datasets`).
- **Tokenizer**: HuggingFace `AutoTokenizer`, padded / truncated to `--max-length` (default 128).

`example/bert/macro_factory.py` walks the module tree and swaps **every `nn.Linear`** — Q/K/V/output attention projections, FFN intermediate / output, the `[CLS]` pooler, and the classifier — for a crossbar-backed counterpart built from `neurox.architecture.unit.cim.CimUnit` plus the QAT operators in `example/bert/quant.py`. Embeddings, LayerNorm, GELU, and the attention softmax stay in float. (NeuroX core stops at the architecture layer; the layer-rewriting walk is example-side code.)

## Extra dependencies

This example needs `transformers` and `datasets` on top of NeuroX's core deps:

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
  --cim_macro physical
```

## Notes

- **Quantization grid**: weight / activation grids match `example/bert/macro.toml` (4-state RRAM × 1-digit weight ⇒ ±3, 2-state × 4-digit activation ⇒ [0, 15]).  The output grid defaults to signed 8-bit (`y_qmin=-128`, `y_qmax=127`) so BERT's post-layer activation range survives requantization.
- **Hardware is TOML-driven**: every circuit parameter — RRAM states, switch, wires, digital datapath, tile geometry, output quantization — comes from the config TOML, and every nonideality switch from the paired `*.policy.toml`; the CLI has no hardware knobs. `example/bert/macro_ideal.toml` and its policy instantiate an ideal xbar directly and load as they stand.  `example/bert/macro.toml` and its policy instead pull the xbar in through `_neurox_use`; the fragment they name is not in the tree, so that pair — the CLI default — stops at config load.
- **Output quantization**: the ideal config declares the maximum converter resolution (`adc_bits`) plus one canonical inclusive window `[lower, upper]`, in MAC units, per quantization mode. A call selects `quantization_mode` and `adc_active_bits`; `adc_active_bits = 0` is the ideal macro's lossless oracle, and the ideal macro is the output-code rescale reference.
- **ADC resolution and depth**: per-tile output quantization compounds across BERT-small's 26 linear layers, so a coarse converter drives the stack to chance accuracy. For deep transformer stacks raise the configured `adc_bits` before running HAT / eval.
- **Knowledge distillation**: a frozen float teacher supervises the HAT student via soft-label KL (temperature `T=2.0`) plus pooled `[CLS]` MSE.  Combined with a high-resolution ADC, one epoch lands at ≈ 83 % val_acc — comfortably above the 80 % bar.  Without KD the same student stalls around chance.
- **HAT cost**: BERT-small has 4 encoder blocks × 6 linears each + pooler + classifier = 26 linear layers, each running through the macro on every forward.  HAT training is much slower than float fine-tuning; start with small batch size (16) and few epochs (1-3).
- **Sequence length**: SST-2 sentences are short (median ≈ 11 tokens); `--max-length 128` is comfortable but you can drop to 64 for speed.
- **TOML consistency between HAT and eval**: train and evaluate with the same config pair so the saved `(multiplier, rshift, bias_int)` buffers align with the evaluator's physical-to-ideal code scale reported by `rescale_factor`.

# BERT-small on SST-2

The evaluation entry observes linear-operator energy while retaining the checkpoint model's numerical outputs. It uses the bundled Xue2020 or Ye2023 configuration with an ideal macro twin.

Run from the NeuroX repository root using its uv environment:

```bash
uv run --no-sync python -m example.bert.evaluate --checkpoint weight/bert_small_float.pth --preset xue2020jssc --device cuda:0 --num-samples 10 --output log/energy/bert_xue.json
```

Each evaluated sample is checked against inference without energy observation.

Checkpoint weights, tokenizer files, and SST-2 examples must be available locally. `--sequence-length` controls token padding and truncation. Embedding, normalization, activation, and attention matrix-product costs are outside the linear/convolution observation scope.

See the [energy evaluation guide](../energy/README.md) for dependency extras, quantization assumptions, preset selection, and the four-project campaign. Ideal-mode energy excludes analog dynamic energy.

# BERT-small linear-layer shapes

Shapes for every `nn.Linear` that NeuroX `replace_for_hat` / `build_evaluator` swaps for a crossbar-backed counterpart when this example runs against `google/bert_uncased_L-4_H-512_A-8` (4 layers, hidden=512, heads=8, intermediate=2048, binary classification head → num_labels=2).

`weight.shape = (out_features, in_features)`.  Tile counts assume the `example/bert/macro.toml` geometry of **64 × 64** (`col_num=64`, `row_num=64`): `Tc = ceil(out/64)`, `Tr = ceil(in/64)`, `tiles = Tc · Tr`.

## Per encoder layer (4 identical blocks, indices 0–3)

| module                   | `(out, in)` |   Tc |   Tr |   tiles |
| ------------------------ | ----------: | ---: | ---: | ------: |
| `attention.self.query`   |  (512, 512) |    8 |    8 |      64 |
| `attention.self.key`     |  (512, 512) |    8 |    8 |      64 |
| `attention.self.value`   |  (512, 512) |    8 |    8 |      64 |
| `attention.output.dense` |  (512, 512) |    8 |    8 |      64 |
| `intermediate.dense`     | (2048, 512) |   32 |    8 |     256 |
| `output.dense`           | (512, 2048) |    8 |   32 |     256 |
| **per-block subtotal**   |             |      |      | **768** |

## Model head

| module              | `(out, in)` |   Tc |   Tr | tiles |
| ------------------- | ----------: | ---: | ---: | ----: |
| `bert.pooler.dense` |  (512, 512) |    8 |    8 |    64 |
| `classifier`        |    (2, 512) |    1 |    8 |     8 |

## Totals

| quantity                    |                                       value |
| --------------------------- | ------------------------------------------: |
| encoder blocks              |                                           4 |
| linear layers total         | **26** (4 blocks × 6 + pooler + classifier) |
| crossbar tiles total        |                4 × 768 + 64 + 8 = **3 144** |
| bias present on every layer |                                         yes |

## Notes

- **FFN layers dominate**: one encoder block's two FFN linears contribute
  512 tiles, twice as many as all four attention projections (256).
- **Classifier is the shortest path**: only 8 tiles map the 512-dim
  pooled `[CLS]` embedding to 2 logits, so its per-layer quantization
  error directly drives final-accuracy sensitivity.
- **Input side (K) is 512 for all layers except the FFN output**
  (2048) — that is the only layer whose 32 row-tiles concatenate in
  the shift-add stage rather than the per-tile ADC stage.

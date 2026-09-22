# BERT-small linear-layer shapes

The `google/bert_uncased_L-4_H-512_A-8` checkpoint has four encoder blocks, hidden width 512, eight attention heads, intermediate width 2048, and a binary classification head. Energy evaluation observes its complete logical linear operators, with weights ordered as `(out_features, in_features)`.

| Operator | Weight shape | Positions per encoder block |
| --- | --- | --- |
| Query, key, and value projections | `(512, 512)` | 3 |
| Attention output projection | `(512, 512)` | 1 |
| FFN input projection | `(2048, 512)` | 1 |
| FFN output projection | `(512, 2048)` | 1 |

The pooler has shape `(512, 512)` and the classifier `(2, 512)`. Biases are present on all these layers. Hardware placement follows the selected preset, slice counts, and input-slot sharing configuration; logical shapes alone do not determine macro counts.

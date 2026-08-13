"""BERT-small architecture for SST-2 binary classification.

A 4-layer, 512-hidden, 8-head encoder wrapped by HuggingFace
`BertForSequenceClassification`, which adds a single
`nn.Linear(hidden, num_labels)` classifier on top of the `[CLS]` pooled
output. Loading goes through the explicit `Bert*` classes: some community
BERT-small checkpoints ship a `config.json` without the `model_type` key the
`Auto*` loader routes on.
"""

import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

DEFAULT_MODEL_NAME: str = "google/bert_uncased_L-4_H-512_A-8"


def create_bert_small(
    num_labels: int = 2,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | None = None,
) -> nn.Module:
    """Build BERT-small with a sequence-classification head.

    Args:
        num_labels: Classifier width; 2 for SST-2 positive / negative.
        model_name: HuggingFace model identifier; must match the tokenizer
            the inputs are built with.
        cache_dir: Cache directory for the pretrained weights.

    Returns:
        A `BertForSequenceClassification` with the pretrained encoder loaded
        and a randomly initialised classifier head.
    """
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    return BertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
    )

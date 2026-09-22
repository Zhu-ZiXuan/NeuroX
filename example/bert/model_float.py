"""Pretrained BERT-small construction for SST-2 classification."""

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
        model_name: HuggingFace model identifier; must match the tokenizer
            the inputs are built with.

    Returns:
        A `BertForSequenceClassification` with the pretrained encoder loaded
        and a randomly initialised classifier head.
    """
    # Concrete BERT classes support checkpoints whose config omits `model_type`.
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    return BertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
    )

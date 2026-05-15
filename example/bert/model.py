"""BERT-small architecture for SST-2 binary classification.

Uses ``prajjwal1/bert-small`` (4 layers, 512 hidden, 8 heads, ~28M
params) wrapped by HuggingFace ``BertForSequenceClassification``,
which adds a single ``nn.Linear(hidden, num_labels)`` classifier on top
of the ``[CLS]`` pooled output.

NeuroX's ``replace_for_hat`` / ``build_evaluator`` walk the module tree
and swap every ``nn.Linear`` for its crossbar-backed counterpart —
which here means the BERT Q/K/V/output projections, the FFN
intermediate / output layers, the pooler, and the classifier.  Other
ops (LayerNorm, GELU, attention softmax, embeddings) stay in float.

Note on ``from_pretrained``: we use the explicit ``Bert*`` classes
(not ``Auto*``) because some community BERT-small checkpoints ship a
``config.json`` without the ``model_type`` key that the ``Auto*``
loader needs to route to the right class.  ``BertForSequenceClassification``
bypasses that lookup.
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
        num_labels: Number of classes for the classifier head.  Default
            ``2`` matches SST-2 (positive / negative sentiment).
        model_name: HuggingFace model identifier.  Default
            ``prajjwal1/bert-small``.
        cache_dir: Optional cache directory for the pretrained weights.

    Returns:
        A ``BertForSequenceClassification`` instance with the pretrained
        BERT-small encoder loaded; only the classifier head is randomly
        initialised and needs fine-tuning.
    """
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    return BertForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        cache_dir=cache_dir,
    )

"""SST-2 (GLUE) data loading + tokenization for the BERT example."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME: str = "google/bert_uncased_L-4_H-512_A-8"


def _tokenize_split(split_name: str, dataset_dir: Path, model_name: str, max_length: int) -> Dataset:
    """Load and tokenize one SST-2 split, cached under `dataset_dir`."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw = load_dataset("nyu-mll/glue", "sst2", split=split_name, cache_dir=str(dataset_dir))
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(dataset_dir))

    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return dict(
            tokenizer(
                batch["sentence"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
        )

    tokenized = raw.map(_tokenize, batched=True, desc=f"Tokenizing SST-2 {split_name}")
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return tokenized


def _collate(batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return input_ids, attention_mask, token_type_ids, labels


def create_sst2_dataloader(
    dataset_dir: Path,
    batch_size: int,
    device: torch.device,
    *,
    split: str = "validation",
    shuffle: bool = False,
    max_length: int = 128,
    model_name: str = DEFAULT_MODEL_NAME,
    indices: Sequence[int] | None = None,
) -> DataLoader:
    """Create an SST-2 dataloader rooted at `dataset_dir`.

    Args:
        dataset_dir: Cache directory for the GLUE / SST-2 corpus and the
            tokenizer files, passed to HuggingFace as `cache_dir`.
        device: Runtime device, consulted only to decide `pin_memory`.
        split: `"train"` selects the training split, anything else the
            validation split. GLUE ships the SST-2 test split unlabelled, so
            scoring runs on validation.
        max_length: Token sequence length every sample is padded or truncated
            to.
        model_name: HuggingFace tokenizer identifier; must match the model it
            feeds so the vocabulary aligns.

    Returns:
        DataLoader yielding 4-tuples
        `(input_ids, attention_mask, token_type_ids, labels)`.
    """
    split_name = "train" if split == "train" else "validation"
    dataset = _tokenize_split(split_name, dataset_dir, model_name, max_length)
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

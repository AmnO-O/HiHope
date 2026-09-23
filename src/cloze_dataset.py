"""Prompt-based Dataset for Cloze Compositionality Probing.

This dataset processes input tabular records (Sentence, Word, Compound, Label)
and constructs tokenized prompt sequences containing a single [MASK] token.
It precisely tracks the index of [MASK] for extraction by ClozeCompositionalityModel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .cloze_prompts import build_cloze_prompt


class ClozePromptDataset(Dataset):
    """Dataset producing tokenized Cloze sequences for Masked Probing."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_token: Optional[str] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token = mask_token or tokenizer.mask_token or "[MASK]"
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids(self.mask_token)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Extract record fields
        sentence = item.get("sentence") or item.get("Sentence") or item.get("context") or ""
        target_name = item.get("target")
        if target_name == "mod":
            word = item.get("mod") or item.get("word") or ""
            label = item.get("mod_avg")
        elif target_name == "head":
            word = item.get("head") or item.get("word") or ""
            label = item.get("head_avg")
        elif target_name == "pv":
            word = item.get("compound") or item.get("mod") or item.get("word") or ""
            label = item.get("mod_avg")  # PV avg stored in mod_avg
        else:
            word = item.get("word") or item.get("Word") or item.get("mod") or item.get("head") or ""
            label = item.get("score") or item.get("rating") or item.get("label") or item.get("mod_avg")

        compound = item.get("compound") or item.get("Compound") or ""
        lang = item.get("lang") or item.get("language") or "en"
        is_pv = bool(item.get("is_pv", False)) or (target_name == "pv")
        filename = item.get("filename") or item.get("file") or ""

        # Construct semantic prompt
        prompt_text, sem_type = build_cloze_prompt(
            sentence=sentence,
            word=word,
            compound=compound,
            lang=lang,
            is_pv=is_pv,
            filename=filename,
            mask_token=self.mask_token,
        )

        # Tokenize prompt sequence
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Locate the exact position of the [MASK] token
        try:
            mask_pos = input_ids.index(self.mask_token_id)
        except ValueError:
            # Fallback if truncated: replace the last non-special token with [MASK]
            mask_pos = max(1, len(input_ids) - 2)
            input_ids[mask_pos] = self.mask_token_id

        # Target rating / label if available
        if label is None or not (isinstance(label, (int, float)) and not (isinstance(label, float) and (label != label))):
            label_val = 0.0
            has_label = False
        else:
            label_val = float(label)
            has_label = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask_index": mask_pos,
            "label": label_val,
            "has_label": has_label,
            "target": target_name or "mod",
            "lang": lang,
            "sem_type": sem_type,
        }


def collate_cloze_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Collates a list of ClozePromptDataset items into padded tensors."""
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    mask_indices = [item["mask_index"] for item in batch]
    labels = [item["label"] for item in batch]
    has_labels = [item["has_label"] for item in batch]
    targets = [item["target"] for item in batch]
    langs = [item["lang"] for item in batch]
    sem_types = [item["sem_type"] for item in batch]

    # Dynamic padding to max sequence length in this batch
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "mask_indices": torch.tensor(mask_indices, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "has_label": torch.tensor(has_labels, dtype=torch.bool),
        "targets": targets,
        "langs": langs,
        "sem_types": sem_types,
    }

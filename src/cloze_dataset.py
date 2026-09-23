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
        prompt_style: str = "score",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_style = prompt_style
        self.mask_token = mask_token or tokenizer.mask_token or "[MASK]"
        self.mask_token_id = tokenizer.mask_token_id or tokenizer.convert_tokens_to_ids(self.mask_token)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # Extract record fields
        sentence = item.get("sentence") if item.get("sentence") is not None else (
            item.get("Sentence") if item.get("Sentence") is not None else item.get("context", "")
        )
        target_name = item.get("target")
        std = None
        if target_name == "mod":
            word = item.get("mod") or item.get("word") or ""
            label = item.get("mod_avg")
            std = item.get("mod_std")
        elif target_name == "head":
            word = item.get("head") or item.get("word") or ""
            label = item.get("head_avg")
            std = item.get("head_std")
        elif target_name == "pv":
            word = item.get("compound") or item.get("mod") or item.get("word") or ""
            label = item.get("mod_avg")  # PV avg stored in mod_avg
            std = item.get("mod_std")
        else:
            word = item.get("word") or item.get("Word") or item.get("mod") or item.get("head") or ""
            label = item.get("score")
            if label is None:
                label = item.get("rating")
            if label is None:
                label = item.get("label")
            if label is None:
                label = item.get("mod_avg")
            std = item.get("mod_std") or item.get("std") or item.get("Std")

        compound = item.get("compound") or item.get("Compound") or ""
        lang = str(item.get("lang") or item.get("language") or "en")
        is_pv = bool(item.get("is_pv", False)) or (target_name == "pv")
        is_aux = bool(item.get("is_aux", False))
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
            style=self.prompt_style,
        )

        # Tokenize prompt sequence with special guarantee that [MASK] is never truncated
        encoding = self.tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        if len(input_ids) > self.max_length:
            # If sequence exceeds max_length, truncate tokens from the context sentence (middle),
            # never from the prompt suffix which contains [MASK]
            try:
                raw_mask_pos = input_ids.index(self.mask_token_id)
                # Keep the suffix containing [MASK] and special ending token
                suffix_len = len(input_ids) - raw_mask_pos + 1
                available_for_prefix = max(10, self.max_length - suffix_len)
                input_ids = input_ids[:available_for_prefix] + input_ids[raw_mask_pos - 1:]
                attention_mask = [1] * len(input_ids)
            except ValueError:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]

        # Locate the exact position of the [MASK] token
        try:
            mask_pos = input_ids.index(self.mask_token_id)
        except ValueError:
            mask_pos = max(1, len(input_ids) - 2)
            input_ids[mask_pos] = self.mask_token_id


        # Target rating / label if available
        if label is None or not (isinstance(label, (int, float)) and not (isinstance(label, float) and (label != label))):
            label_val = 0.0
            has_label = False
        else:
            label_val = float(label)
            has_label = True

        std_val = float(std) if std is not None and isinstance(std, (int, float)) and not (isinstance(std, float) and std != std) else float("nan")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask_index": mask_pos,
            "label": label_val,
            "std": std_val,
            "has_label": has_label,
            "target": target_name or "mod",
            "lang": lang,
            "is_aux": is_aux,
            "sem_type": sem_type,
        }


def collate_cloze_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, Any]:
    """Collates a list of ClozePromptDataset items into padded tensors.
    
    All dictionary fields are returned as torch.Tensor to ensure batch.items() .to(device)
    never crashes on string lists.
    """
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    mask_indices = [item["mask_index"] for item in batch]
    labels = [item["label"] for item in batch]
    stds = [item["std"] for item in batch]
    has_labels = [item["has_label"] for item in batch]
    is_auxs = [item["is_aux"] for item in batch]

    # Target codes: 0 = mod, 1 = head, 2 = pv
    target_map = {"mod": 0, "head": 1, "pv": 2}
    target_codes = [target_map.get(item["target"], 0) for item in batch]

    # Lang codes: 0 = en, 1 = de
    lang_codes = [1 if str(item["lang"]).lower().startswith("de") else 0 for item in batch]

    # Semantic type codes: 0 = compound, 1 = bare_lemma, 2 = particle_verb
    sem_map = {"compound": 0, "bare_lemma": 1, "particle_verb": 2}
    sem_codes = [sem_map.get(item["sem_type"], 0) for item in batch]

    # Dynamic padding to max sequence length in this batch
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    stds_tensor = torch.tensor(stds, dtype=torch.float32)
    has_labels_tensor = torch.tensor(has_labels, dtype=torch.bool)
    is_auxs_tensor = torch.tensor(is_auxs, dtype=torch.bool)
    target_tensor = torch.tensor(target_codes, dtype=torch.long)
    is_pv_tensor = (target_tensor == 2)

    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "mask_indices": torch.tensor(mask_indices, dtype=torch.long),
        "labels": labels_tensor,
        "stds": stds_tensor,
        "has_label": has_labels_tensor,
        "is_aux": is_auxs_tensor,
        "target": target_tensor,
        "lang_code": torch.tensor(lang_codes, dtype=torch.long),
        "sem_type_code": torch.tensor(sem_codes, dtype=torch.long),
        # Compatibility aliases for two-stream loss and metric collectors
        "mod_avg": labels_tensor,
        "head_avg": labels_tensor,
        "mod_std": stds_tensor,
        "head_std": stds_tensor,
        "is_pv": is_pv_tensor,
    }


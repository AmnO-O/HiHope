"""Cloze-Based Compositionality Probing Model using Masked Language Model Hidden States.

This architecture leverages the pre-trained cloze-answering capabilities of mmBERT:
    1. Cross-Attention: All tokens of the context sentence, the target word,
       and the compound cross-attend bidirectionally across all layers.
    2. Multi-Layer Extraction: In accordance with Miletić & Schulte im Walde (2023, 2025),
       early/middle layers (Layers 4-8) capture lexical semantics while higher layers
       (Layers 14-18) capture contextual syntax.
    3. Verbalizer Prior: Projects logits at [MASK] for literal vs figurative tokens
       to provide an inductive zero-shot grounding score Delta_verb.
    4. Continuous Head: GaussHead(h_mask, Delta_verb) predicts (mu, sigma) with NLL + CCC loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .cloze_prompts import VERBALIZER_TOKENS
from .heads import GaussHead


class MultiLayerAggregator(nn.Module):
    """Softmax-weighted aggregation of intermediate hidden states."""

    def __init__(self, layer_indices: List[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.weights = nn.Parameter(torch.zeros(len(layer_indices)))

    def forward(self, all_hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Args: all_hidden_states tuple of [B, L, H].
        Returns weighted mean representation [B, L, H].
        """
        selected = [all_hidden_states[idx] for idx in self.layer_indices if idx < len(all_hidden_states)]
        if not selected:
            return all_hidden_states[-1]
        stacked = torch.stack(selected, dim=0)  # [K, B, L, H]
        norm_weights = F.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * norm_weights).sum(dim=0)


class ClozeCompositionalityModel(nn.Module):
    """Cloze-Prompt Masked Probing Model for Compositionality Prediction."""

    def __init__(
        self,
        model_name_or_path: str = "distilbert/distilbert-base-multilingual-cased",
        extract_layers: Optional[List[int]] = None,
        head_hidden: int = 256,
        dropout: float = 0.1,
        sigma_floor: float = 0.04,
        use_verbalizer_prior: bool = True,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__()
        self.model_name = model_name_or_path
        self.use_verbalizer_prior = use_verbalizer_prior

        # Load pre-trained Masked LM. The checkpoint ships untied
        # embeddings + LM-head weights, so silence HF's "both present, will
        # NOT tie" warning by declaring tie_word_embeddings=False.
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        model_config.tie_word_embeddings = False
        self.mlm = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            config=model_config,
            output_hidden_states=True,
        )
        hidden_size = getattr(self.mlm.config, "hidden_size", 768)

        # Multi-layer aggregator (Default: early-semantic layer 6, mid layer 14, upper layer 20)
        self.extract_layers = extract_layers or [6, 14, 20]
        self.layer_agg = MultiLayerAggregator(self.extract_layers)

        # Tokenizer & Verbalizer token ID resolution
        self.tokenizer = tokenizer
        self.verbalizer_ids: Dict[str, Dict[str, List[int]]] = {}
        if self.tokenizer is not None and use_verbalizer_prior:
            self._cache_verbalizer_ids()

        # Prediction Head: receives h_mask (768) + verbalizer prior scalar (1) if enabled
        feature_in_dim = hidden_size + (1 if use_verbalizer_prior else 0)
        self.head = GaussHead(
            in_features=feature_in_dim,
            hidden=head_hidden,
            dropout=dropout,
            floor=sigma_floor,
        )

    def _cache_verbalizer_ids(self) -> None:
        """Cache tokenizer token ID sequences for literal and figurative verbalizer vocabulary."""
        assert self.tokenizer is not None
        for lang, words_dict in VERBALIZER_TOKENS.items():
            self.verbalizer_ids[lang] = {"literal": [], "figurative": []}
            for polarity in ["literal", "figurative"]:
                for word in words_dict[polarity]:
                    ids = self.tokenizer.encode(word, add_special_tokens=False)
                    if ids:
                        # Store all subword token ids for full coverage
                        self.verbalizer_ids[lang][polarity].extend(ids)
                # Deduplicate
                self.verbalizer_ids[lang][polarity] = sorted(list(set(self.verbalizer_ids[lang][polarity])))

    def compute_verbalizer_prior(
        self,
        mask_logits: torch.Tensor,
        lang_list: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute the verbalizer logit difference via logsumexp over candidate tokens:
        Delta_verb = logsumexp(logits[literal_tokens]) - logsumexp(logits[figurative_tokens])
        
        Args:
            mask_logits: [B, VocabSize] MLM prediction logits at the [MASK] position.
            lang_list: List of language codes for each sample in the batch.
        
        Returns:
            [B, 1] scaled scalar prior.
        """
        batch_size = mask_logits.size(0)
        if not self.verbalizer_ids:
            return torch.zeros(batch_size, 1, device=mask_logits.device, dtype=mask_logits.dtype)

        diffs = []
        for i in range(batch_size):
            lang = lang_list[i] if lang_list and i < len(lang_list) else "en"
            lang_key = "de" if str(lang).lower().startswith("de") else "en"
            lit_ids = self.verbalizer_ids.get(lang_key, {}).get("literal", [])
            fig_ids = self.verbalizer_ids.get(lang_key, {}).get("figurative", [])

            if lit_ids and fig_ids:
                lit_logit = torch.logsumexp(mask_logits[i, lit_ids], dim=-1)
                fig_logit = torch.logsumexp(mask_logits[i, fig_ids], dim=-1)
                diffs.append(lit_logit - fig_logit)
            else:
                diffs.append(torch.tensor(0.0, device=mask_logits.device, dtype=mask_logits.dtype))

        delta = torch.stack(diffs, dim=0).unsqueeze(-1)
        # Normalize with tanh to keep features nicely bounded between -1.0 and 1.0
        return torch.tanh(delta)


    def forward(
        self,
        batch_or_input_ids: Union[Dict[str, Any], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.Tensor] = None,
        lang: Optional[List[str]] = None,
        with_logits: bool = False,
        with_pv: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass for prompt cloze probing.
        
        Supports both direct tensor inputs and batch dictionary input.
        """
        if isinstance(batch_or_input_ids, dict):
            input_ids = batch_or_input_ids["input_ids"]
            attention_mask = batch_or_input_ids["attention_mask"]
            mask_indices = batch_or_input_ids["mask_indices"]
            if "lang_code" in batch_or_input_ids:
                code_tensor = batch_or_input_ids["lang_code"]
                lang = ["de" if int(c) == 1 else "en" for c in code_tensor.cpu().tolist()]
            elif "langs" in batch_or_input_ids:
                lang = batch_or_input_ids["langs"]
            else:
                lang = None
        else:
            input_ids = batch_or_input_ids

        outputs = self.mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 1. Multi-layer hidden state aggregation across semantic and syntax layers
        aggregated_hidden = self.layer_agg(outputs.hidden_states)  # [B, SeqLen, Hidden]

        # 2. Extract hidden state at the exact [MASK] position
        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        h_mask = aggregated_hidden[batch_indices, mask_indices]  # [B, Hidden]

        # 3. Optional Verbalizer Logit Prior computation
        if self.use_verbalizer_prior and hasattr(outputs, "logits"):
            mask_logits = outputs.logits[batch_indices, mask_indices]  # [B, VocabSize]
            delta_verb = self.compute_verbalizer_prior(mask_logits, lang_list=lang)
            feature_vector = torch.cat([h_mask, delta_verb], dim=-1)
        else:
            feature_vector = h_mask

        # 4. Predict continuous Gaussian distribution parameters
        mu, sigma = self.head(feature_vector)
        pred = mu.squeeze(-1) if mu.dim() > 1 else mu
        sig = sigma.squeeze(-1) if sigma.dim() > 1 else sigma

        if with_logits:
            # GaussLoss expects 6-tuple: (mod, head, pv, mod_logits, head_logits, pv_logits)
            # where logits channel holds predicted sigma
            if with_pv:
                return pred, pred, pred, sig, sig, sig
            return pred, pred, sig, sig

        if with_pv:
            # In single-target cloze evaluation, pred acts as the active target pred
            return pred, pred, pred
        return pred, pred

    def pred_heads(self) -> List[nn.Module]:
        """Return the prediction heads and layer weights for Phase 1 unfreezing."""
        modules: List[nn.Module] = [self.head]
        if self.layer_agg is not None:
            modules.append(self.layer_agg)
        return modules

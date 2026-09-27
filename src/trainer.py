"""Trainer for Target-Aware Query Attention compositionality model."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import Config
from src.data import CompDataset, collate_comp, expand_targets
from src.losses import GaussLoss
from src.model import build_model
from src.train import evaluate, train_epoch, unfreeze_top_layers


@dataclass
class FoldResult:
    fold: Optional[int]
    rho_mod: float
    rho_head: float
    rho_mean: float
    best_epoch: int
    ckpt_path: str
    history: List[Dict]
    best_mod_pred: np.ndarray
    best_head_pred: np.ndarray
    best_mod_label: np.ndarray
    best_head_label: np.ndarray
    rho_pv: float = 0.0


class Trainer:
    """Trains the Target-Aware Query Attention & Component Interaction model."""

    def __init__(self, cfg: Config, device, logger: logging.Logger, output_dir: Path):
        self.cfg = cfg
        self.device = torch.device(device)
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_loaders(self, train_rows: List[Dict], val_rows: List[Dict], tokenizer):
        active_targets = self.cfg.targets
        if active_targets:
            train_rows = expand_targets(train_rows, active_targets)
            val_rows = expand_targets(val_rows, active_targets)

        train_ds = CompDataset(train_rows, tokenizer, max_len=self.cfg.max_context_length)
        val_ds = CompDataset(val_rows, tokenizer, max_len=self.cfg.max_context_length)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            collate_fn=collate_comp, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.eval_batch_size, shuffle=False,
            collate_fn=collate_comp, pin_memory=torch.cuda.is_available()
        )
        return train_loader, val_loader

    def fit(
        self,
        train_rows: List[Dict],
        val_rows: List[Dict],
        tokenizer = None,
        fold: Optional[int] = None,
        ckpt_name: Optional[str] = None,
        load_from: Optional[str | Path] = None,
        **kwargs,
    ) -> FoldResult:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.backbone, trust_remote_code=True)
        train_loader, val_loader = self._build_loaders(train_rows, val_rows, tokenizer)

        model = build_model(self.cfg, self.device, load_from=load_from or getattr(self.cfg, 'load_from', None))
        criterion = GaussLoss(
            ccc_weight=self.cfg.ccc_weight,
            ccc_var_floor=self.cfg.ccc_var_floor,
            bin_sigma=self.cfg.bin_sigma,
            use_label_std=self.cfg.use_label_std,
            kl_weight=self.cfg.kl_weight,
        ).to(self.device)

        scaler = torch.amp.GradScaler('cuda') if (self.cfg.fp16 and torch.cuda.is_available()) else None

        # --- Phase 1: Train query attention heads & role embeddings (backbone frozen) ---
        for param in model.lm.parameters():
            param.requires_grad = False

        head_params = [p for m in model.pred_heads() for p in m.parameters() if p.requires_grad]
        optimizer = AdamW(head_params, lr=self.cfg.lr_heads, weight_decay=self.cfg.weight_decay)

        total_steps = len(train_loader) * (self.cfg.freeze_epochs + self.cfg.unfreeze_epochs)
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_rho = -1.0
        best_epoch = 0
        tag = f"_fold_{fold}" if fold is not None else ""
        best_ckpt = self.output_dir / (ckpt_name if ckpt_name else f'best_model{tag}.pt')
        history: List[Dict] = []
        best_preds = None

        self.logger.info("Starting Phase 1 (Frozen Backbone): %d epochs", self.cfg.freeze_epochs)
        for epoch in range(1, self.cfg.freeze_epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler, self.device,
                grad_clip=self.cfg.max_grad_norm, accum_steps=self.cfg.gradient_accumulation_steps
            )
            rho_mod, rho_head, rho_pv, rho_mean, m_preds, h_preds, m_lbls, h_lbls = evaluate(
                model, val_loader, self.device
            )
            self.logger.info(
                "Epoch %02d (Freeze) - Loss: %.4f | Val Rho Mod: %.4f, Head: %.4f, PV: %.4f, Mean: %.4f",
                epoch, train_loss, rho_mod, rho_head, rho_pv, rho_mean
            )
            history.append({'epoch': epoch, 'phase': 'freeze', 'train_loss': train_loss, 'rho_mean': rho_mean})

            if rho_mean > best_rho:
                best_rho = rho_mean
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt)
                best_preds = (rho_mod, rho_head, rho_pv, m_preds, h_preds, m_lbls, h_lbls)

        # --- Phase 2: Unfreeze top layers ---
        unfreeze_n = getattr(self.cfg, 'unfreeze_layers', 4)
        self.logger.info("Starting Phase 2 (Fine-Tuning %s Layers): %d epochs", "All" if unfreeze_n < 0 else f"Top-{unfreeze_n}", self.cfg.unfreeze_epochs)
        unfreeze_top_layers(model, num_layers=unfreeze_n)
        
        backbone_params = [p for p in model.lm.parameters() if p.requires_grad]
        optimizer = AdamW([
            {'params': head_params, 'lr': self.cfg.lr_heads * 0.5},
            {'params': backbone_params, 'lr': self.cfg.lr_backbone},
        ], weight_decay=self.cfg.weight_decay)

        p2_steps = len(train_loader) * self.cfg.unfreeze_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(p2_steps * self.cfg.warmup_ratio), num_training_steps=p2_steps)

        for epoch in range(self.cfg.freeze_epochs + 1, self.cfg.freeze_epochs + self.cfg.unfreeze_epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler, self.device,
                grad_clip=self.cfg.max_grad_norm, accum_steps=self.cfg.gradient_accumulation_steps
            )
            rho_mod, rho_head, rho_pv, rho_mean, m_preds, h_preds, m_lbls, h_lbls = evaluate(
                model, val_loader, self.device
            )
            self.logger.info(
                "Epoch %02d (Fine-Tune) - Loss: %.4f | Val Rho Mod: %.4f, Head: %.4f, PV: %.4f, Mean: %.4f",
                epoch, train_loss, rho_mod, rho_head, rho_pv, rho_mean
            )
            history.append({'epoch': epoch, 'phase': 'unfreeze', 'train_loss': train_loss, 'rho_mean': rho_mean})

            if rho_mean > best_rho:
                best_rho = rho_mean
                best_epoch = epoch
                torch.save(model.state_dict(), best_ckpt)
                best_preds = (rho_mod, rho_head, rho_pv, m_preds, h_preds, m_lbls, h_lbls)

        if best_preds is None:
            best_preds = (0.0, 0.0, 0.0, np.array([]), np.array([]), np.array([]), np.array([]))

        # Save final epoch checkpoint
        final_ckpt = self.output_dir / f'final_model{tag}.pt'
        torch.save(model.state_dict(), final_ckpt)
        self.logger.info("Saved final epoch checkpoint to %s", final_ckpt)

        # Save training history and summary
        summary_path = self.output_dir / f'training_summary{tag}.json'
        summary_dict = {
            'fold': fold,
            'best_epoch': best_epoch,
            'best_rho_mean': float(best_rho),
            'best_rho_mod': float(best_preds[0]),
            'best_rho_head': float(best_preds[1]),
            'best_rho_pv': float(best_preds[2]),
            'best_ckpt': str(best_ckpt),
            'final_ckpt': str(final_ckpt),
            'history': history,
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2)

        # Also provide alias copies for seamless compatibility across scripts
        if fold == 0 or fold is None:
            torch.save(model.state_dict(), self.output_dir / 'final_model.pt')
            if best_ckpt.exists():
                torch.save(torch.load(best_ckpt, weights_only=True), self.output_dir / 'best_model.pt')
                torch.save(torch.load(best_ckpt, weights_only=True), self.output_dir / 'best_model_fold_0.pt')

        return FoldResult(
            fold=fold,
            rho_mod=best_preds[0],
            rho_head=best_preds[1],
            rho_pv=best_preds[2],
            rho_mean=best_rho,
            best_epoch=best_epoch,
            ckpt_path=str(best_ckpt),
            history=history,
            best_mod_pred=best_preds[3],
            best_head_pred=best_preds[4],
            best_mod_label=best_preds[5],
            best_head_label=best_preds[6],
        )

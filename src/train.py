"""Training and evaluation routines for Target-Aware Query Attention."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from .losses import GaussLoss, ccc_loss
from .utils import get_logger

logger = get_logger('src.train')


def _safe_rho(y: np.ndarray, p: np.ndarray) -> float:
    """Calculate Spearman correlation safely."""
    if len(y) < 2 or len(p) < 2:
        return 0.0
    r = float(spearmanr(y, p).statistic)
    return 0.0 if np.isnan(r) or np.isinf(r) else r


def unfreeze_top_layers(model: nn.Module, num_layers: int = 4) -> None:
    """Unfreeze the top N transformer layers of the backbone."""
    encoder = getattr(model.lm, 'encoder', model.lm)
    layers = getattr(encoder, 'layer', None) or getattr(encoder, 'layers', None)
    if layers is not None:
        total = len(layers)
        for i in range(max(0, total - num_layers), total):
            for param in layers[i].parameters():
                param.requires_grad = True


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    criterion: GaussLoss,
    scaler,
    device,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
) -> float:
    """One training epoch for Target-Aware Query Attention."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        allowed = batch.get('has_label', torch.ones(batch['input_ids'].size(0), dtype=torch.bool, device=device))
        is_pv = batch.get('is_pv', torch.zeros_like(allowed, dtype=torch.bool))
        allowed_nn = allowed & (~is_pv)
        allowed_pv = allowed & is_pv

        if 'target' in batch:
            tgt = batch['target']
            mod_mask = allowed_nn & (tgt == 0)
            head_mask = allowed_nn & (tgt == 1)
            pv_mask = allowed_pv & (tgt == 2)
        else:
            mod_mask = head_mask = allowed_nn
            pv_mask = allowed_pv

        with torch.autocast(device_type=device_type, enabled=(scaler is not None)):
            mod_pred, head_pred, pv_pred, mod_sigma, head_sigma, pv_sigma = model(
                batch, with_logits=True, with_pv=True
            )

            mod_loss = criterion(mod_pred, batch['mod_avg'], mod_sigma, std=batch.get('mod_std'), mask=mod_mask)
            head_loss = criterion(head_pred, batch['head_avg'], head_sigma, std=batch.get('head_std'), mask=head_mask)
            pv_loss = criterion(pv_pred, batch['mod_avg'], pv_sigma, std=batch.get('mod_std'), mask=pv_mask)
            
            loss = (mod_loss + head_loss + pv_loss) / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step_idx % accum_steps == 0 or step_idx == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on validation/trial set."""
    model.eval()
    all_mod_preds, all_head_preds, all_pv_preds = [], [], []
    all_mod_labels, all_head_labels, all_pv_labels = [], [], []
    all_is_pv = []

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        mod_pred, head_pred, pv_pred = model(batch, with_logits=False, with_pv=True)

        all_mod_preds.append(mod_pred.cpu().numpy())
        all_head_preds.append(head_pred.cpu().numpy())
        all_pv_preds.append(pv_pred.cpu().numpy())
        all_mod_labels.append(batch['mod_avg'].cpu().numpy())
        all_head_labels.append(batch['head_avg'].cpu().numpy())
        all_is_pv.append(batch['is_pv'].cpu().numpy())

    mod_preds = np.concatenate(all_mod_preds)
    head_preds = np.concatenate(all_head_preds)
    pv_preds = np.concatenate(all_pv_preds)
    mod_labels = np.concatenate(all_mod_labels)
    head_labels = np.concatenate(all_head_labels)
    is_pv = np.concatenate(all_is_pv)

    nn_mask = (~is_pv) & np.isfinite(mod_labels) & np.isfinite(head_labels)
    pv_mask = is_pv & np.isfinite(mod_labels)

    rho_mod = _safe_rho(mod_labels[nn_mask], mod_preds[nn_mask]) if nn_mask.any() else 0.0
    rho_head = _safe_rho(head_labels[nn_mask], head_preds[nn_mask]) if nn_mask.any() else 0.0
    rho_pv = _safe_rho(mod_labels[pv_mask], pv_preds[pv_mask]) if pv_mask.any() else 0.0
    rho_mean = (rho_mod + rho_head) / 2.0

    return rho_mod, rho_head, rho_pv, rho_mean, mod_preds, head_preds, mod_labels, head_labels

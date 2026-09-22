"""Owns the full gauss scoring fit loop for one split/fold."""

from __future__ import annotations

import gc
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from transformers import get_constant_schedule, get_linear_schedule_with_warmup

from src.config import Config
from src.data import CompDataset, collate_comp, expand_targets
from src.losses import GaussLoss
from src.model import apply_lora, build_model, lora_parameters
from src.train import evaluate, track_optimizer_steps, train_epoch, unfreeze_top_layers
from src.utils import resolve_paths


def _safe_rho(y: np.ndarray, p: np.ndarray) -> float:
    r = float(spearmanr(y, p).statistic) if len(y) > 1 else 0.0
    return 0.0 if r != r else r


def _embeddings(model) -> nn.Module:
    """Input-embedding module of the plain AutoModel backbone."""
    return model.lm.get_input_embeddings()


class WeightEMA:
    """Shadow-copy exponential moving average over the trainable parameters.

    ``update`` is called once per real optimizer step; ``apply_to``/``restore``
    temporarily load the averaged weights into the live module so validation,
    checkpointing and ``best`` predictions all come from the denoised model.
    Params that are frozen at any given time (backbone, phase-1 LoRA) are
    tracked but never blended, and begin averaging from their current value the
    moment they become trainable (phase-2 unfreeze). Shadows live on CPU so the
    EMA never consumes the scarce device memory.
    """

    def __init__(self, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.raw: Dict[str, torch.Tensor] = {}
        self._params: List[tuple] = []
        self._applied = False

    def register(self, model) -> None:
        self._params = list(model.named_parameters())
        for name, p in self._params:
            self.shadow[name] = p.detach().cpu().clone()

    def _active(self) -> List[tuple]:
        return [(n, p) for n, p in self._params if p.requires_grad]

    def update(self, model) -> None:
        if not self._params:
            self.register(model)
        with torch.no_grad():
            d = self.decay
            for name, p in self._active():
                self.shadow[name].mul_(d).add_(p.detach().cpu(), alpha=1 - d)

    def apply_to(self, model) -> None:
        if self._applied:
            return
        self.raw.clear()
        for name, p in self._active():
            self.raw[name] = p.detach().cpu().clone()
            p.data.copy_(self.shadow[name])
        self._applied = True

    def restore(self, model) -> None:
        if not self._applied:
            return
        for name, p in self._active():
            p.data.copy_(self.raw[name])
        self.raw.clear()
        self._applied = False


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
    def __init__(self, cfg: Config, device, logger: logging.Logger, output_dir: Path):
        self.cfg = cfg
        self.device = torch.device(device)
        self.logger = logger
        self.output_dir = Path(output_dir)
        self._val_rows: List[Dict] = []

    # ------------------------------------------------------------------ #
    def _build_loaders(self, train_rows, val_rows, tokenizer):
        self.logger.info("Building Datasets & Tokenizing %d train / %d val rows...", len(train_rows), len(val_rows))
        if self.cfg.targets:
            train_rows = expand_targets(train_rows, self.cfg.targets)
            val_rows = expand_targets(val_rows, self.cfg.targets)
            self.logger.info('Single-target mode: expanded to %d train / %d val rows (targets=%s)',
                             len(train_rows), len(val_rows), self.cfg.targets)
        self._val_rows = list(val_rows)

        train_ds = CompDataset(
            train_rows, tokenizer, max_len=self.cfg.max_context_length,
            proto_stream=self.cfg.proto_stream,
            target_mask_prob=getattr(self.cfg, 'target_mask_prob', 0.0),
            proto_mode=getattr(self.cfg, 'proto_mode', 'hybrid'),
            max_proto_length=getattr(self.cfg, 'max_proto_length', 32))
        val_ds = CompDataset(
            val_rows, tokenizer, max_len=self.cfg.max_context_length,
            proto_stream=self.cfg.proto_stream,
            target_mask_prob=0.0,
            proto_mode=getattr(self.cfg, 'proto_mode', 'hybrid'),
            max_proto_length=getattr(self.cfg, 'max_proto_length', 32))
        
        # Chỉ bật persistent_workers khi num_workers > 0 để tránh deadlock
        num_workers = max(0, self.cfg.num_workers)
        use_workers = num_workers > 0
        is_cuda = (getattr(self.device, 'type', '') == 'cuda')
        
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=is_cuda,
            persistent_workers=use_workers, collate_fn=collate_comp)

        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=is_cuda,
            persistent_workers=use_workers, collate_fn=collate_comp)
            
        self.logger.info("DataLoaders ready (num_workers=%d, pin_memory=%s)", num_workers, is_cuda)
        return train_loader, val_loader

    def _apply_lora(self, model):
        if not self.cfg.lora_targets:
            return []
        adapters = apply_lora(
            model, rank=self.cfg.lora_rank, alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout, targets=self.cfg.lora_targets,
            from_layer=self.cfg.lora_from_layer)
        paths = getattr(model, '_lora_paths', [])
        wtgt = getattr(model, '_lora_targets_used', None)
        self.logger.info('Applied LoRA: %d adapters (r=%d, alpha=%d, layers >= %d) e.g. %s%s',
                         len(adapters), self.cfg.lora_rank, self.cfg.lora_alpha,
                         self.cfg.lora_from_layer, paths[:3],
                         f' (auto-fell back to targets {wtgt})' if wtgt else '')
        return adapters

    # ------------------------------------------------------------------ #
    def _pred_heads(self, model) -> List[nn.Module]:
        """The prediction heads, fusion modules, and interaction layers."""
        if hasattr(model, 'pred_heads'):
            return model.pred_heads()
        heads: List[nn.Module] = []
        for attr in ('fusion', 'head', 'mod_gauss', 'head_gauss', 'pv_gauss', 'shift_fuse'):
            m = getattr(model, attr, None)
            if m is not None and isinstance(m, nn.Module):
                heads.append(m)
        seen = set()
        uniq: List[nn.Module] = []
        for m in heads:
            if id(m) not in seen:
                seen.add(id(m))
                uniq.append(m)
        return uniq

    # ------------------------------------------------------------------ #
    def _param_groups(self, model, adapters, phase: int):
        emb = list(_embeddings(model).parameters())
        heads = self._pred_heads(model)
        head = []
        for m in heads:
            head += list(m.parameters())
        head_ids = {id(p) for p in head}
        emb_ids = {id(p) for p in emb}
        others = [p for n, p in model.named_parameters()
                  if p.requires_grad and id(p) not in head_ids and id(p) not in emb_ids]

        groups = []
        if self.cfg.embedding_lr > 0 and emb:
            groups.append({'params': emb, 'lr': self.cfg.embedding_lr,
                           'weight_decay': 0.0, 'tag': 'frozen'})
        groups.append({'params': head, 'lr': self.cfg.head_lr,
                       'weight_decay': self.cfg.weight_decay, 'tag': 'head'})
        if others:
            groups.append({'params': others, 'lr': self.cfg.encoder_lr,
                           'weight_decay': self.cfg.weight_decay, 'tag': 'encoder'})
        self.logger.info('Phase %d param groups: %s',
                         phase, [len(g['params']) for g in groups])
        return groups

    def _freeze_phase1(self, model, adapters):
        """Phase 1: freeze backbone and LoRA, train only prediction heads."""
        for p in model.parameters():
            p.requires_grad = False
        for m in self._pred_heads(model):
            for p in m.parameters():
                p.requires_grad = True
        if self.cfg.embedding_lr > 0:
            for p in _embeddings(model).parameters():
                p.requires_grad = True
        for p in lora_parameters(adapters):
            p.requires_grad = False

    def _unfreeze_phase2(self, model, adapters):
        """Phase 2: unfreeze LoRA adapters and optionally top encoder layers."""
        for p in lora_parameters(adapters):
            p.requires_grad = True
        if self.cfg.unfreeze_from_layer > 0:
            unfreeze_top_layers(model, self.cfg.unfreeze_from_layer)

    def _optimizer(self, model, adapters, phase, steps):
        groups = self._param_groups(model, adapters, phase)
        optimizer = AdamW(groups)
        track_optimizer_steps(optimizer)
        n_steps = int(max(1, steps))
        if phase == 1:
            sched_type = getattr(self.cfg, 'head_lr_schedule', 'constant')
            min_ratio = float(getattr(self.cfg, 'head_lr_min_ratio', 0.1))
            warmup_steps = int(n_steps * float(getattr(self.cfg, 'warmup_ratio', 0.0)))
            if sched_type == 'cosine':
                def lr_lambda_fn(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, n_steps - warmup_steps))
                    progress = min(max(progress, 0.0), 1.0)
                    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
                scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
            elif sched_type == 'linear':
                def lr_lambda_fn(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    progress = float(step - warmup_steps) / float(max(1, n_steps - warmup_steps))
                    progress = min(max(progress, 0.0), 1.0)
                    return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
                scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
            else:
                scheduler = get_constant_schedule(optimizer)
        else:
            # Phase 2 directly from epoch 0:
            sched_type = getattr(self.cfg, 'head_lr_schedule', 'cosine')
            min_ratio = float(getattr(self.cfg, 'head_lr_min_ratio', 0.1))

            def make_head_lambda():
                if sched_type == 'cosine':
                    return lambda step: min_ratio + (1.0 - min_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * min(max(float(step) / float(max(1, n_steps)), 0.0), 1.0))
                    )
                elif sched_type == 'linear':
                    return lambda step: min_ratio + (1.0 - min_ratio) * (
                        1.0 - min(max(float(step) / float(max(1, n_steps)), 0.0), 1.0)
                    )
                return lambda step: 1.0

            def make_encoder_lambda():
                return lambda step: 0.5 * (
                    1.0 + math.cos(math.pi * min(max(float(step) / float(max(1, n_steps)), 0.0), 1.0))
                )

            lr_lambda = [
                make_head_lambda() if g.get('tag', 'encoder') == 'head'
                else (lambda step: 1.0) if g.get('tag', 'encoder') == 'frozen'
                else make_encoder_lambda()
                for g in groups
            ]
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    # ------------------------------------------------------------------ #
    def fit(self, train_rows, val_rows, tokenizer, fold: Optional[int] = None,
            ckpt_name: str = 'best.pt', load_from: Optional[str | Path] = None) -> FoldResult:
        
        train_loader, val_loader = self._build_loaders(train_rows, val_rows, tokenizer)

        self.logger.info("Building model architecture (load_from=%s)...", load_from)
        model = build_model(self.cfg, self.device, load_from=load_from)
        adapters = self._apply_lora(model)
        ema = WeightEMA(self.cfg.ema_decay) if self.cfg.ema_decay > 0 else None
        if ema is not None:
            ema.register(model)
            self.logger.info('EMA on: tracking %d params (decay=%.4f)',
                             len(ema.shadow), self.cfg.ema_decay)

        steps_per_epoch = math.ceil(len(train_loader) / self.cfg.accum_steps)
        if self.cfg.freeze_epochs > 0:
            self._freeze_phase1(model, adapters)
            phase_init = 1
            steps_init = steps_per_epoch * self.cfg.freeze_epochs
        else:
            self._freeze_phase1(model, adapters)
            self._unfreeze_phase2(model, adapters)
            phase_init = 2
            steps_init = steps_per_epoch * self.cfg.total_epochs

        optimizer, scheduler = self._optimizer(
            model, adapters, phase=phase_init, steps=steps_init)

        scaler = GradScaler(
            'cuda',
            enabled=(self.device.type == 'cuda'),
            init_scale=self.cfg.amp_init_scale,
            growth_interval=self.cfg.amp_growth_interval,
        )
        criterion = GaussLoss(
            ccc_weight=self.cfg.ccc_weight,
            ccc_var_floor=self.cfg.ccc_var_floor,
            bin_sigma=self.cfg.bin_sigma, use_label_std=self.cfg.use_label_std,
            kl_weight=self.cfg.kl_weight,
        ).to(self.device)

        best_rho, best_epoch, no_improve_epochs = -float('inf'), -1, 0
        best: Optional[Dict[str, np.ndarray]] = None
        history: List[Dict] = []
        ckpt_dir = self.output_dir / 'models'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / ckpt_name

        self.logger.info("Starting training loop for %d epochs...", self.cfg.total_epochs)

        for epoch in range(self.cfg.total_epochs):
            if epoch == self.cfg.freeze_epochs and self.cfg.freeze_epochs > 0:
                self.logger.info(
                    '>>> Entering Phase 2 (unfreezing LoRA%s) at epoch %d <<<',
                    f', top layers from {self.cfg.unfreeze_from_layer}' if self.cfg.unfreeze_from_layer > 0 else '',
                    epoch + 1)
                
                self._unfreeze_phase2(model, adapters)
                
                # Preserve existing head optimizer state and add newly unfrozen LoRA/encoder parameters
                existing_pids = {id(p) for g in optimizer.param_groups for p in g['params']}
                new_encoder_params = [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and id(p) not in existing_pids
                ]
                if new_encoder_params:
                    optimizer.add_param_group({
                        'params': new_encoder_params,
                        'lr': self.cfg.encoder_lr,
                        'initial_lr': self.cfg.encoder_lr,
                        'weight_decay': self.cfg.weight_decay,
                        'tag': 'encoder'
                    })
                    self.logger.info(
                        'Preserved head optimizer states and added %d newly unfrozen parameters to optimizer.',
                        len(new_encoder_params))
                
                # Reset base LRs cleanly for Phase 2
                for g in optimizer.param_groups:
                    tag = g.get('tag', 'encoder')
                    if tag == 'head':
                        g['lr'] = self.cfg.head_lr
                        g['initial_lr'] = self.cfg.head_lr
                    elif tag == 'frozen':
                        g['lr'] = self.cfg.embedding_lr
                        g['initial_lr'] = self.cfg.embedding_lr
                    else:
                        g['lr'] = self.cfg.encoder_lr
                        g['initial_lr'] = self.cfg.encoder_lr
                
                remaining_steps = int(max(1, steps_per_epoch * self.cfg.lora_epochs))
                sched_type = getattr(self.cfg, 'head_lr_schedule', 'cosine')
                min_ratio = float(getattr(self.cfg, 'head_lr_min_ratio', 0.1))

                def make_head_lambda():
                    if sched_type == 'cosine':
                        return lambda step: min_ratio + (1.0 - min_ratio) * 0.5 * (
                            1.0 + math.cos(math.pi * min(max(float(step) / float(max(1, remaining_steps)), 0.0), 1.0))
                        )
                    elif sched_type == 'linear':
                        return lambda step: min_ratio + (1.0 - min_ratio) * (
                            1.0 - min(max(float(step) / float(max(1, remaining_steps)), 0.0), 1.0)
                        )
                    return lambda step: 1.0

                def make_encoder_lambda():
                    return lambda step: 0.5 * (
                        1.0 + math.cos(math.pi * min(max(float(step) / float(max(1, remaining_steps)), 0.0), 1.0))
                    )

                lr_lambda = [
                    make_head_lambda() if g.get('tag', 'encoder') == 'head'
                    else (lambda step: 1.0) if g.get('tag', 'encoder') == 'frozen'
                    else make_encoder_lambda()
                    for g in optimizer.param_groups
                ]
                scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
                no_improve_epochs = 0   # fresh patience window for the LoRA phase

            phase = 'FROZEN' if epoch < self.cfg.freeze_epochs else 'UNFROZEN-TOP'
            diag: Dict[str, float] = {}
            
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler,
                self.device, grad_clip=self.cfg.grad_clip,
                accum_steps=self.cfg.accum_steps, report=diag,
                ema=ema,
                proto_rank_loss_weight=getattr(self.cfg, 'proto_rank_loss', 0.0),
                proto_margin=getattr(self.cfg, 'proto_margin', 0.2),
                aux_loss_weight=getattr(self.cfg, 'aux_loss_weight', 1.0),
                use_wep_infonce=getattr(self.cfg, 'use_wep_infonce', False),
                wep_tau=getattr(self.cfg, 'wep_tau', 0.10),
                wep_weight=getattr(self.cfg, 'wep_weight', 0.08),
                wep_use_std_attenuation=getattr(self.cfg, 'wep_use_std_attenuation', False),
            )

            # Compute train rho directly from in-epoch predictions on CORE rows (~is_aux)
            # to provide an apples-to-apples comparison with Val rho
            if 'train_preds' in diag:
                preds_tuple = diag['train_preds']
                if len(preds_tuple) == 10:
                    tr_m, tr_h, tr_p, tr_my, tr_hy, tr_py, tr_al, tr_is_pv, tr_is_aux, tr_tgt = preds_tuple
                    tr_core = ~tr_is_aux.astype(bool)
                    tr_al_m = tr_al & (tr_tgt == 0) & tr_core
                    tr_al_h = tr_al & (tr_tgt == 1) & tr_core
                    tr_al_p = tr_al & (tr_tgt == 2) & tr_core
                elif len(preds_tuple) == 9:
                    tr_m, tr_h, tr_p, tr_my, tr_hy, tr_py, tr_al, tr_is_pv, tr_is_aux = preds_tuple
                    tr_is_pv = tr_is_pv.astype(bool)
                    tr_core = ~tr_is_aux.astype(bool)
                    tr_al_m = tr_al_h = tr_al & (~tr_is_pv) & tr_core
                    tr_al_p = tr_al & tr_is_pv & tr_core
                elif len(preds_tuple) == 8:
                    tr_m, tr_h, tr_p, tr_my, tr_hy, tr_py, tr_al, tr_is_pv = preds_tuple
                    tr_is_pv = tr_is_pv.astype(bool)
                    tr_al_m = tr_al_h = tr_al & (~tr_is_pv)
                    tr_al_p = tr_al & tr_is_pv
                else:
                    tr_al_m = tr_al_h = tr_al_p = np.array([False])
                tr_rho_m = _safe_rho(tr_my[tr_al_m], tr_m[tr_al_m]) if tr_al_m.any() else 0.0
                tr_rho_h = _safe_rho(tr_hy[tr_al_h], tr_h[tr_al_h]) if tr_al_h.any() else 0.0
                tr_rho_pv = _safe_rho(tr_py[tr_al_p], tr_p[tr_al_p]) if tr_al_p.any() else 0.0
            else:
                tr_rho_m = tr_rho_h = tr_rho_pv = 0.0
                tr_al_m = tr_al_h = tr_al_p = np.array([], dtype=bool)
            tr_nn_any = tr_al_m.any() or tr_al_h.any()
            tr_pv_any = tr_al_p.any()
            if tr_pv_any and tr_nn_any:
                train_rho_mean = (tr_rho_m + tr_rho_h + tr_rho_pv) / 3.0
            elif tr_pv_any:
                train_rho_mean = tr_rho_pv
            else:
                train_rho_mean = (tr_rho_m + tr_rho_h) / 2.0

            if ema is not None:
                ema.apply_to(model)
            try:
                val_mod, val_head, val_pv, val_mod_y, val_head_y, val_mask, val_diag = evaluate(
                    model, val_loader, self.device, return_all=True, return_pv=True, return_diagnostics=True)
            finally:
                if ema is not None:
                    ema.restore(model)

            is_pv_mask = np.array([bool(r.get('is_pv', False)) for r in self._val_rows])
            nn_mask = val_mask & (~is_pv_mask)
            pv_mask = val_mask & is_pv_mask

            if self.cfg.targets:
                # Single-target mode: each head only has ground truth on the
                # rows that routed to it, so restrict each rho to its target.
                trg = np.array([0 if r.get('target') == 'mod'
                                else (1 if r.get('target') == 'head'
                                      else 2) for r in self._val_rows])
                nn_mod_mask = val_mask & (~is_pv_mask) & (trg == 0)
                nn_head_mask = val_mask & (~is_pv_mask) & (trg == 1)
                pv_mask = val_mask & is_pv_mask & (trg == 2)
            else:
                nn_mod_mask = nn_head_mask = nn_mask

            rho_mod = _safe_rho(val_mod_y[nn_mod_mask], val_mod[nn_mod_mask]) if nn_mod_mask.any() else 0.0
            rho_head = _safe_rho(val_head_y[nn_head_mask], val_head[nn_head_mask]) if nn_head_mask.any() else 0.0

            rho_pv = _safe_rho(val_mod_y[pv_mask], val_pv[pv_mask]) if pv_mask.any() else 0.0

            # Per-language PV rho for the aux ablation: de-pv comes from the
            # competition dev split, en-pv from Cordeiro/NCTTI/litnlit aux.
            lang_arr = np.array([str(r.get('lang', '')) for r in self._val_rows])
            de_pv_mask = pv_mask & (lang_arr == 'de')
            en_pv_mask = pv_mask & (lang_arr == 'en')
            rho_pv_de = _safe_rho(val_mod_y[de_pv_mask], val_pv[de_pv_mask]) if de_pv_mask.any() else 0.0
            rho_pv_en = _safe_rho(val_mod_y[en_pv_mask], val_pv[en_pv_mask]) if en_pv_mask.any() else 0.0

            if pv_mask.any() and nn_mask.any():
                rho_mean = (rho_mod + rho_head + rho_pv) / 3.0
            elif pv_mask.any():
                rho_mean = rho_pv
            else:
                rho_mean = (rho_mod + rho_head) / 2.0

            # --- Diagnostics: Raw Prototype-Context Cosine vs Gold Labels ---
            val_mod_cos = val_diag['mod_cos']
            val_head_cos = val_diag['head_cos']
            val_pv_cos = val_diag['pv_cos']

            cos_rho_mod = _safe_rho(val_mod_y[nn_mod_mask], val_mod_cos[nn_mod_mask]) if nn_mod_mask.any() else 0.0
            cos_rho_head = _safe_rho(val_head_y[nn_head_mask], val_head_cos[nn_head_mask]) if nn_head_mask.any() else 0.0
            cos_rho_pv = _safe_rho(val_mod_y[pv_mask], val_pv_cos[pv_mask]) if pv_mask.any() else 0.0

            all_eval_cos = []
            all_eval_y = []
            if nn_mod_mask.any():
                all_eval_cos.append(val_mod_cos[nn_mod_mask])
                all_eval_y.append(val_mod_y[nn_mod_mask])
            if nn_head_mask.any() and not self.cfg.targets:
                all_eval_cos.append(val_head_cos[nn_head_mask])
                all_eval_y.append(val_head_y[nn_head_mask])
            if pv_mask.any():
                all_eval_cos.append(val_pv_cos[pv_mask])
                all_eval_y.append(val_mod_y[pv_mask])

            if all_eval_cos:
                pooled_cos = np.concatenate(all_eval_cos)
                pooled_y = np.concatenate(all_eval_y)
                cos_rho_pooled = _safe_rho(pooled_y, pooled_cos)
                cos_min, cos_max, cos_avg = float(pooled_cos.min()), float(pooled_cos.max()), float(pooled_cos.mean())
            else:
                cos_rho_pooled, cos_min, cos_max, cos_avg = 0.0, 0.0, 0.0, 0.0

            # --- Diagnostics: Reliance Gate Aggregation over (row, exit) per state ---
            gate_stats = None
            if val_diag.get('mod_gate') is not None:
                mod_g = val_diag['mod_gate']
                head_g = val_diag['head_gate']
                pv_g = val_diag['pv_gate']
                ast = val_diag['align_state']

                gate_by_state = {0: [], 1: [], 2: []}
                for i in range(len(ast)):
                    if not val_mask[i]:
                        continue
                    s = int(ast[i])
                    if s in gate_by_state:
                        if is_pv_mask[i]:
                            gate_by_state[s].append(float(pv_g[i]))
                        else:
                            if self.cfg.targets:
                                if trg[i] == 0:
                                    gate_by_state[s].append(float(mod_g[i]))
                                elif trg[i] == 1:
                                    gate_by_state[s].append(float(head_g[i]))
                                else:
                                    gate_by_state[s].append(float(pv_g[i]))
                            else:
                                gate_by_state[s].append(float(mod_g[i]))
                                gate_by_state[s].append(float(head_g[i]))

                def _fmt_stat(vals):
                    arr = np.array(vals)
                    if len(arr) == 0:
                        return "N/A"
                    return f"{arr.mean():.2f} ± {arr.std():.2f}"

                clean_str = _fmt_stat(gate_by_state[0])
                degen_str = _fmt_stat(gate_by_state[1])
                fallback_str = _fmt_stat(gate_by_state[2])
                gate_stats = (clean_str, len(gate_by_state[0]),
                              degen_str, len(gate_by_state[1]),
                              fallback_str, len(gate_by_state[2]))

            ovf_str = ''
            lr_str = f"lr {diag['lr']:.2e}"
            if 'encoder_lr' in diag:
                lr_str += f" (enc {diag['encoder_lr']:.2e})"

            if pv_mask.any():
                self.logger.info(
                    'Epoch %d/%d [%s] | Loss %.4f (nn_m %.4f / nn_h %.4f / pv %.4f) | '
                    'Train Mod ρ %.4f | Train Head ρ %.4f | Train PV ρ %.4f | Train Mean ρ %.4f | '
                    'Val Mod ρ %.4f | Val Head ρ %.4f | Val PV ρ %.4f (de %.4f / en %.4f) | Val Mean ρ %.4f | '
                    'steps %d (skip %d) | scale %.1f | %s',
                    epoch + 1, self.cfg.total_epochs, phase, train_loss,
                    diag.get('nn_mod_loss', diag['mod_loss']),
                    diag.get('nn_head_loss', diag['head_loss']),
                    diag.get('pv_loss', 0.0),
                    tr_rho_m, tr_rho_h, tr_rho_pv, train_rho_mean,
                    rho_mod, rho_head, rho_pv, rho_pv_de, rho_pv_en, rho_mean,
                    diag['opt_steps'], diag['skipped'], diag['scale'], lr_str)
            else:
                self.logger.info(
                    'Epoch %d/%d [%s] | Loss %.4f (mod %.4f / head %.4f) | '
                    'Train Mod ρ %.4f | Train Head ρ %.4f | Train Mean ρ %.4f | '
                    'Val Mod ρ %.4f | Val Head ρ %.4f | Val Mean ρ %.4f | '
                    'steps %d (skip %d) | scale %.1f | %s',
                    epoch + 1, self.cfg.total_epochs, phase, train_loss,
                    diag['mod_loss'], diag['head_loss'],
                    tr_rho_m, tr_rho_h, train_rho_mean,
                    rho_mod, rho_head, rho_mean,
                    diag['opt_steps'], diag['skipped'], diag['scale'], lr_str)

            # Log diagnostic signals
            if pv_mask.any():
                self.logger.info(
                    '  [Diag] Raw CosSim ρ (h): mod %.4f | head %.4f | pv %.4f | pooled %.4f (range [%.2f, %.2f], mean %.2f)',
                    cos_rho_mod, cos_rho_head, cos_rho_pv, cos_rho_pooled, cos_min, cos_max, cos_avg)
            else:
                self.logger.info(
                    '  [Diag] Raw CosSim ρ (h): mod %.4f | head %.4f | pooled %.4f (range [%.2f, %.2f], mean %.2f)',
                    cos_rho_mod, cos_rho_head, cos_rho_pooled, cos_min, cos_max, cos_avg)

            # Projected Metric Cosine vs Gold Labels
            val_mod_cos_z = val_diag.get('mod_cos_z')
            val_head_cos_z = val_diag.get('head_cos_z')
            val_pv_cos_z = val_diag.get('pv_cos_z')
            if val_mod_cos_z is not None and len(val_mod_cos_z) > 0 and (val_mod_cos_z != 0).any():
                cos_z_rho_mod = _safe_rho(val_mod_y[nn_mod_mask], val_mod_cos_z[nn_mod_mask]) if nn_mod_mask.any() else 0.0
                cos_z_rho_head = _safe_rho(val_head_y[nn_head_mask], val_head_cos_z[nn_head_mask]) if nn_head_mask.any() else 0.0
                cos_z_rho_pv = _safe_rho(val_mod_y[pv_mask], val_pv_cos_z[pv_mask]) if pv_mask.any() else 0.0

                all_eval_cos_z = []
                if nn_mod_mask.any():
                    all_eval_cos_z.append(val_mod_cos_z[nn_mod_mask])
                if nn_head_mask.any() and not self.cfg.targets:
                    all_eval_cos_z.append(val_head_cos_z[nn_head_mask])
                if pv_mask.any():
                    all_eval_cos_z.append(val_pv_cos_z[pv_mask])
                if all_eval_cos_z:
                    pooled_cos_z = np.concatenate(all_eval_cos_z)
                    cos_z_rho_pooled = _safe_rho(pooled_y, pooled_cos_z)
                else:
                    cos_z_rho_pooled = 0.0

                if pv_mask.any():
                    self.logger.info(
                        '  [Diag] Proj CosSim ρ (z): mod %.4f | head %.4f | pv %.4f | pooled %.4f',
                        cos_z_rho_mod, cos_z_rho_head, cos_z_rho_pv, cos_z_rho_pooled)
                else:
                    self.logger.info(
                        '  [Diag] Proj CosSim ρ (z): mod %.4f | head %.4f | pooled %.4f',
                        cos_z_rho_mod, cos_z_rho_head, cos_z_rho_pooled)

            if gate_stats is not None:
                self.logger.info(
                    '  [Diag] Gate: clean g=%s (n=%d) | degen g=%s (n=%d) | fallback g=%s (n=%d)',
                    gate_stats[0], gate_stats[1],
                    gate_stats[2], gate_stats[3],
                    gate_stats[4], gate_stats[5])

            history.append({
                'epoch': epoch + 1, 'phase': phase,
                'loss': round(float(train_loss), 5),
                'loss_mod': round(float(diag.get('nn_mod_loss', diag['mod_loss'])), 5),
                'loss_head': round(float(diag.get('nn_head_loss', diag['head_loss'])), 5),
                'loss_pv': round(float(diag.get('pv_loss', 0.0)), 5),
                'train_rho_mod': round(tr_rho_m, 5),
                'train_rho_head': round(tr_rho_h, 5),
                'train_rho_pv': round(tr_rho_pv, 5),
                'train_rho_mean': round(train_rho_mean, 5),
                'rho_mod': round(rho_mod, 5), 'rho_head': round(rho_head, 5),
                'rho_pv': round(rho_pv, 5),
                'rho_pv_de': round(rho_pv_de, 5), 'rho_pv_en': round(rho_pv_en, 5),
                'rho_mean': round(rho_mean, 5),
                'cos_rho_mod': round(float(cos_rho_mod), 5),
                'cos_rho_head': round(float(cos_rho_head), 5),
                'cos_rho_pv': round(float(cos_rho_pv), 5),
                'cos_rho_pooled': round(float(cos_rho_pooled), 5),
                'opt_steps': diag['opt_steps'], 'skipped': diag['skipped'],
                'grad_norm': round(diag['grad_norm'], 4),
                'scale': round(diag['scale'], 1), 'lr': diag['lr'],
            })

            if rho_mean > best_rho:
                best_rho, best_epoch, no_improve_epochs = rho_mean, epoch + 1, 0
                if self.cfg.targets:
                    sel_mod = val_mask & (np.array([r.get('target') == 'mod' for r in self._val_rows]) if self._val_rows else np.zeros(len(val_mod), dtype=bool))
                    sel_head = val_mask & (np.array([r.get('target') == 'head' for r in self._val_rows]) if self._val_rows else np.zeros(len(val_mod), dtype=bool))
                    sel_pv = val_mask & (np.array([r.get('target') == 'pv' for r in self._val_rows]) if self._val_rows else np.zeros(len(val_mod), dtype=bool))
                else:
                    sel_mod = sel_head = val_mask & (~is_pv_mask)
                    sel_pv = val_mask & is_pv_mask
                best = {
                    'mod': val_mod[sel_mod].copy() if sel_mod.any() else np.array([]),
                    'head': val_head[sel_head].copy() if sel_head.any() else np.array([]),
                    'mod_y': val_mod_y[sel_mod].copy() if sel_mod.any() else np.array([]),
                    'head_y': val_head_y[sel_head].copy() if sel_head.any() else np.array([]),
                    'rho_mod': rho_mod, 'rho_head': rho_head, 'rho_pv': rho_pv,
                }
                if ema is not None:
                    ema.apply_to(model)
                torch.save(model.state_dict(), ckpt_path)
                layer_agg = getattr(model, 'layer_agg', None)
                if layer_agg is not None:
                    w = torch.softmax(layer_agg.weights, dim=0).detach().cpu().numpy()
                    layers = self.cfg.extract_layers or tuple(range(len(w)))
                    w_str = ', '.join(f'L{l}:{v:.3f}' for l, v in zip(layers, w))
                    self.logger.info('Learned Layer Weights at Best Epoch %d: [%s]', epoch + 1, w_str)
                if ema is not None:
                    ema.restore(model)
            else:
                # Early stop only in phase 2 (unfrozen LoRA) so the frozen phase
                # never aborts before the backbone has a chance to adapt;
                # in pure-frozen mode (lora_epochs == 0 or empty lora_targets),
                # early stop is active throughout.
                if epoch >= self.cfg.freeze_epochs or self.cfg.lora_epochs == 0 or not self.cfg.lora_targets:
                    no_improve_epochs += 1
                    if no_improve_epochs >= self.cfg.patience:
                        self.logger.info(
                            'Early stop at epoch %d (best epoch %d, Mean ρ %.4f)',
                            epoch + 1, best_epoch, best_rho)
                        break

        if best is None:
            raise RuntimeError('No improvement over any epoch - check the hyperparameters.')
            
        # Dọn dẹp GPU sạch đống rác sau khi fit xong 1 fold
        del model, optimizer, scheduler, criterion
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        best_rho_mod = best['rho_mod']
        best_rho_head = best['rho_head']
        best_rho_pv = best.get('rho_pv', 0.0)
        best_rho_mean = best_rho
        self.logger.info('Split %s finished | best Mean ρ %.4f (Mod %.4f / Head %.4f / PV %.4f) at epoch %d',
                         'n/a' if fold is None else fold, best_rho_mean, best_rho_mod, best_rho_head, best_rho_pv, best_epoch)

        return FoldResult(
            fold=fold, rho_mod=best_rho_mod, rho_head=best_rho_head,
            rho_mean=best_rho_mean, best_epoch=best_epoch, ckpt_path=str(ckpt_path),
            history=history, best_mod_pred=best['mod'], best_head_pred=best['head'],
            best_mod_label=best['mod_y'], best_head_label=best['head_y'],
            rho_pv=best_rho_pv,
        )

"""Training primitives for the gauss-only pipeline.

  * ``allowed`` row mask = has_label only. Every labeled row trains:
    span-pooled rows use their aligned constituent; degenerate German fused
    compounds (mod/head on ONE token -- the whole compound word) pool that
    position's vector for both roles, and unaligned (missing-span) rows fall
    back to whole-sentence pooling in ``pool_active_context``. In all cases
    the mod/head split is carried by the Stream-1 prototype.
  * ``unfreeze_top_layers`` walks the encoder generically and skips
    ``.linear.`` paths so LoRA base weights stay frozen.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .utils import get_logger

logger = get_logger('src.train')


def track_optimizer_steps(optimizer) -> None:
    """Version-independent 'did the optimizer really step' detector.

    Counts actual ``optimizer.step()`` calls via ``register_step_post_hook``
    (AMP's GradScaler skips the call entirely on overflow, so a real call ==
    a real parameter update). Do NOT rewrite ``optimizer.step``: torch >= 2.6
    rebuilds it when a scheduler is constructed (`patch_track_step_called`).
    """
    if getattr(optimizer, '_cmp_step_counter', None) is not None:
        return
    counter = [0]
    optimizer._cmp_step_counter = counter
    try:
        optimizer.register_step_post_hook(lambda opt, args, kwargs: counter.__setitem__(0, counter[0] + 1))
    except AttributeError:
        optimizer._cmp_step_counter = None


def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device,
                grad_clip=1.0, accum_steps=1, report=None, ema=None,
                proto_rank_loss_weight=0.0, proto_margin=0.2,
                aux_loss_weight=1.0,
                use_wep_infonce=False, wep_tau=0.10, wep_weight=0.08,
                wep_use_std_attenuation=False,
                phase0_only=False, supervised_loss_weight=1.0):
    """One scoring epoch with AMP + gradient accumulation + clipping.

    Returns the mean supervised loss of the epoch.
    """
    if not hasattr(optimizer, '_cmp_step_counter'):
        track_optimizer_steps(optimizer)
    step_counter = getattr(optimizer, '_cmp_step_counter', None)

    # When WEP-InfoNCE is active, disable margin rank loss to prevent gradient interference
    if use_wep_infonce:
        proto_rank_loss_weight = 0.0

    model.train()
    total_loss = 0.0
    mod_loss_sum = head_loss_sum = pv_loss_sum = 0.0
    optimizer.zero_grad()
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    opt_steps = skipped = 0
    last_grad_norm = last_scale = last_lr = float('nan')
    n_micro = len(dataloader)

    tr_mod_preds, tr_head_preds, tr_pv_preds = [], [], []
    tr_mod_targets, tr_head_targets, tr_pv_label = [], [], []
    tr_allowed, tr_is_pv, tr_is_aux = [], [], []
    tr_targets = []

    for step_idx, batch in enumerate(dataloader, 1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Degenerate rows (fused German compound = one token; mod and head share
        # that position's vector, which IS the whole compound word) and unaligned
        # rows (whole-sentence fallback in pool_active_context) both train: the
        # mod/head distinction still reaches the model through the prototype
        # stream (Stream 1). Only rows without labels are excluded.
        allowed = batch['has_label']
        is_pv = batch.get('is_pv', torch.zeros_like(allowed))
        allowed_nn = allowed & (~is_pv)
        allowed_pv = allowed & is_pv

        # Single-target mode: each row answers its own target, so each loss
        # only sees the rows whose active target matches (others are zeroed by
        # the model and would otherwise drag the loss through inactive heads).
        if 'target' in batch:
            tgt = batch['target']
            mod_mask = allowed_nn & (tgt == 0)
            head_mask = allowed_nn & (tgt == 1)
            pv_mask = allowed_pv & (tgt == 2)
        else:
            mod_mask = head_mask = allowed_nn
            pv_mask = allowed_pv

        with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
            requires_logits = getattr(criterion, 'requires_logits', False)

            mod_logits = head_logits = pv_logits = None

            # aux rows act as weak regularization: scale their supervised loss
            # contribution by aux_loss_weight (default 1.0 -> unchanged).
            aux_w = torch.where(
                batch.get('is_aux', torch.zeros_like(allowed)),
                torch.full_like(allowed, max(float(aux_loss_weight), 1e-3), dtype=allowed.dtype),
                torch.ones_like(allowed, dtype=allowed.dtype),
            )

            if requires_logits:
                (mod_pred, head_pred, pv_pred, mod_logits, head_logits, pv_logits) = model(
                    batch, with_logits=True, with_pv=True)
            else:
                mod_pred, head_pred, pv_pred = model(batch, with_pv=True)

            if phase0_only or float(supervised_loss_weight) == 0.0:
                mod_loss = head_loss = pv_loss = torch.tensor(0.0, device=device)
                loss = torch.tensor(0.0, device=device)
            else:
                # NN loss (mask=allowed on NN rows)
                mod_loss = criterion(
                    mod_pred, batch['mod_avg'], mod_logits, batch['mod_std'],
                    mask=mod_mask, weight=aux_w,
                )
                head_loss = criterion(
                    head_pred, batch['head_avg'], head_logits, batch['head_std'],
                    mask=head_mask, weight=aux_w,
                )

                # PV has only an overall Avg/Std label. Its dedicated composition
                # exit consumes both Base and Particle spans; mod/head exits do not
                # receive PV supervision.
                pv_loss = criterion(
                    pv_pred, batch['mod_avg'], pv_logits, batch['mod_std'],
                    mask=pv_mask, weight=aux_w,
                )

                loss = (mod_loss + head_loss + pv_loss) * float(supervised_loss_weight)

            # Within-Exit Partitioned InfoNCE (WEP-InfoNCE)
            if use_wep_infonce and wep_weight > 0.0:
                from .losses import within_exit_infonce_loss
                if 'target' in batch:
                    # Single-target mode
                    last_z_ctx = getattr(model, 'last_z_ctx', None)
                    last_z_proto = getattr(model, 'last_z_proto', None)
                    if last_z_ctx is not None and last_z_proto is not None:
                        tgt = batch['target']
                        labels = torch.where(tgt == 1, batch['head_avg'], batch['mod_avg'])
                        std_labels = None
                        if 'head_std' in batch and 'mod_std' in batch:
                            std_labels = torch.where(tgt == 1, batch['head_std'], batch['mod_std'])
                        wep_loss = within_exit_infonce_loss(
                            z_ctx=last_z_ctx,
                            z_proto=last_z_proto,
                            ratings=labels,
                            has_label=allowed,
                            exit_ids=tgt,
                            tau=wep_tau,
                            exclude_pv=False,
                            std_ratings=std_labels,
                            use_std_attenuation=wep_use_std_attenuation,
                        )
                        loss = loss + wep_weight * wep_loss
                else:
                    # Joint mode: gather mod, head, and pv representations into exit-partitioned batch
                    z_ctx_m = getattr(model, 'last_z_ctx_mod', None)
                    z_proto_m = getattr(model, 'last_z_proto_mod', None)
                    z_ctx_h = getattr(model, 'last_z_ctx_head', None)
                    z_proto_h = getattr(model, 'last_z_proto_head', None)
                    z_ctx_p = getattr(model, 'last_z_ctx_pv', None)
                    z_proto_p = getattr(model, 'last_z_proto_pv', None)
                    if z_ctx_m is not None and z_proto_m is not None and z_ctx_h is not None and z_proto_h is not None:
                        z_c_list = [z_ctx_m, z_ctx_h]
                        z_p_list = [z_proto_m, z_proto_h]
                        r_list = [batch['mod_avg'], batch['head_avg']]
                        has_m = allowed_nn & torch.isfinite(batch['mod_avg'])
                        has_h = allowed_nn & torch.isfinite(batch['head_avg'])
                        hl_list = [has_m, has_h]
                        bsz = z_ctx_m.size(0)
                        e_list = [
                            torch.zeros(bsz, dtype=torch.long, device=device),
                            torch.ones(bsz, dtype=torch.long, device=device),
                        ]
                        s_list = []
                        if 'mod_std' in batch and 'head_std' in batch:
                            s_list = [batch['mod_std'], batch['head_std']]

                        # Exit 2: Particle Verbs (PV)
                        if z_ctx_p is not None and z_proto_p is not None:
                            z_c_list.append(z_ctx_p)
                            z_p_list.append(z_proto_p)
                            r_list.append(batch['mod_avg'])
                            has_p = allowed_pv & torch.isfinite(batch['mod_avg'])
                            hl_list.append(has_p)
                            e_list.append(torch.full((bsz,), 2, dtype=torch.long, device=device))
                            if 'mod_std' in batch:
                                s_list.append(batch['mod_std'])

                        z_c = torch.cat(z_c_list, dim=0)
                        z_p = torch.cat(z_p_list, dim=0)
                        r = torch.cat(r_list, dim=0)
                        hl = torch.cat(hl_list, dim=0)
                        e_ids = torch.cat(e_list, dim=0)
                        s = torch.cat(s_list, dim=0) if s_list else None

                        wep_loss = within_exit_infonce_loss(
                            z_ctx=z_c,
                            z_proto=z_p,
                            ratings=r,
                            has_label=hl,
                            exit_ids=e_ids,
                            tau=wep_tau,
                            exclude_pv=False,
                            std_ratings=s,
                            use_std_attenuation=wep_use_std_attenuation,
                        )
                        loss = loss + wep_weight * wep_loss

            # Contrastive Prototype Margin Ranking Loss
            if proto_rank_loss_weight > 0.0:
                from .prototype_stream import prototype_rank_loss
                if 'target' in batch:
                    last_cos = getattr(model, 'last_cos_sim', None)
                    if last_cos is not None:
                        labels = torch.where(batch['target'] == 1, batch['head_avg'], batch['mod_avg'])
                        rank_loss = prototype_rank_loss(
                            last_cos, labels, margin=proto_margin, mask=allowed
                        )
                        loss = loss + proto_rank_loss_weight * rank_loss
                else:
                    # Joint mode: apply margin ranking loss to each target independently
                    rank_loss = 0.0
                    m_cos = getattr(model, 'last_mod_cos', None)
                    if m_cos is not None and mod_mask.any():
                        rank_loss = rank_loss + prototype_rank_loss(
                            m_cos, batch['mod_avg'], margin=proto_margin, mask=mod_mask
                        )
                    h_cos = getattr(model, 'last_head_cos', None)
                    if h_cos is not None and head_mask.any():
                        rank_loss = rank_loss + prototype_rank_loss(
                            h_cos, batch['head_avg'], margin=proto_margin, mask=head_mask
                        )
                    p_cos = getattr(model, 'last_pv_cos', None)
                    if p_cos is not None and pv_mask.any():
                        rank_loss = rank_loss + prototype_rank_loss(
                            p_cos, batch['mod_avg'], margin=proto_margin, mask=pv_mask
                        )
                    loss = loss + proto_rank_loss_weight * rank_loss

            if not loss.requires_grad:
                loss = loss + (mod_pred.sum() + head_pred.sum() + pv_pred.sum()) * 0.0

            loss = loss / accum_steps

        scaler.scale(loss).backward()

        # Collect train predictions directly to avoid re-evaluating on train_loader
        tr_mod_preds.append(mod_pred.detach().cpu())
        tr_head_preds.append(head_pred.detach().cpu())
        tr_pv_preds.append(pv_pred.detach().cpu())
        tr_mod_targets.append(batch['mod_avg'].cpu())
        tr_head_targets.append(batch['head_avg'].cpu())
        tr_pv_label.append(batch['mod_avg'].cpu())   # PV reports on the compound Avg
        tr_allowed.append(allowed.cpu())
        tr_is_pv.append(is_pv.cpu())
        tr_is_aux.append(batch.get('is_aux', torch.zeros_like(allowed, dtype=torch.bool)).cpu())
        if 'target' in batch:
            tr_targets.append(batch['target'].cpu())

        if step_idx % accum_steps == 0 or step_idx == n_micro:
            scaler.unscale_(optimizer)
            trainable = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
            if trainable:
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip))
            else:
                last_grad_norm = 0.0

            did_step = False
            if step_counter is not None:
                prev = step_counter[0]
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step_counter[0] > prev:
                    scheduler.step()
                    opt_steps += 1
                    did_step = True
                else:
                    skipped += 1
            else:
                prev = getattr(optimizer, '_step_count', None)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if getattr(optimizer, '_step_count', None) != prev:
                    scheduler.step()
                    opt_steps += 1
                    did_step = True
                else:
                    skipped += 1

            last_scale = float(scaler.get_scale())
            last_lrs = [float(x) for x in scheduler.get_last_lr()]
            # Extract head and encoder LRs by group tag
            head_lr_val, enc_lr_val = None, None
            for idx, g in enumerate(optimizer.param_groups):
                tag = g.get('tag', 'encoder')
                lr_val = last_lrs[idx] if idx < len(last_lrs) else float(g['lr'])
                if tag == 'head':
                    head_lr_val = lr_val
                elif tag == 'encoder':
                    enc_lr_val = lr_val
            last_lr = head_lr_val if head_lr_val is not None else (last_lrs[0] if last_lrs else 0.0)

            if did_step and ema is not None:
                ema.update(model)

        v = loss.item() * accum_steps
        total_loss += v if math.isfinite(v) else 0.0
        mv = mod_loss.item() * accum_steps
        hv = head_loss.item() * accum_steps
        pvv = pv_loss.item() * accum_steps
        mod_loss_sum += mv if math.isfinite(mv) else 0.0
        head_loss_sum += hv if math.isfinite(hv) else 0.0
        pv_loss_sum += pvv if math.isfinite(pvv) else 0.0

    if report is not None:
        rep = {
            'opt_steps': opt_steps, 'skipped': skipped,
            'grad_norm': last_grad_norm,
            'scale': last_scale, 'lr': last_lr,
            'nn_mod_loss': mod_loss_sum / n_micro,
            'nn_head_loss': head_loss_sum / n_micro,
            'pv_loss': pv_loss_sum / n_micro,
            'mod_loss': mod_loss_sum / n_micro,
            'head_loss': head_loss_sum / n_micro,
        }
        if enc_lr_val is not None:
            rep['encoder_lr'] = enc_lr_val
        report.update(rep)
        if tr_mod_preds:
            m_p = torch.cat(tr_mod_preds).numpy()
            h_p = torch.cat(tr_head_preds).numpy()
            p_p = torch.cat(tr_pv_preds).numpy()
            m_y = torch.cat(tr_mod_targets).numpy()
            h_y = torch.cat(tr_head_targets).numpy()
            py = torch.cat(tr_pv_label).numpy()
            al = torch.cat(tr_allowed).numpy()
            ip = torch.cat(tr_is_pv).numpy()
            ia = torch.cat(tr_is_aux).numpy()
            if tr_targets:
                t_t = torch.cat(tr_targets).numpy()
                report['train_preds'] = (m_p, h_p, p_p, m_y, h_y, py, al, ip, ia, t_t)
            else:
                report['train_preds'] = (m_p, h_p, p_p, m_y, h_y, py, al, ip, ia)
    return total_loss / n_micro



def evaluate(model, dataloader, device, return_all: bool = False,
             return_pv: bool = False, return_diagnostics: bool = False):
    """Predictions + gold labels + optional diagnostics.

    If return_all=True: returns (all_mod, all_head, all_mod_y, all_head_y, label_mask)
    for all rows regardless of whether has_label is True or False.
    If return_all=False: returns (mod[mask], head[mask], mod_y[mask], head_y[mask]) if labeled,
    or (mod, head).
    If return_diagnostics=True: appends a diagnostics dict as the final tuple element.
    """
    model.eval()
    all_mod, all_head, all_pv = [], [], []
    all_mod_y, all_head_y = [], []
    masks, has_mask = [], False
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    # Diagnostics accumulation (zero overhead unless requested)
    all_mod_cos, all_head_cos, all_pv_cos = [], [], []
    all_mod_cos_z, all_head_cos_z, all_pv_cos_z = [], [], []
    all_align_state = []
    all_mod_gate, all_head_gate, all_pv_gate = [], [], []
    has_gate = False

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type, enabled=(device_type == 'cuda')):
                if return_pv:
                    mod_pred, head_pred, pv_pred = model(batch, with_pv=True)
                else:
                    mod_pred, head_pred = model(batch)

            all_mod.append(mod_pred.detach().cpu().numpy().reshape(-1))
            all_head.append(head_pred.detach().cpu().numpy().reshape(-1))
            if return_pv:
                all_pv.append(pv_pred.detach().cpu().numpy().reshape(-1))

            if return_diagnostics:
                mc = getattr(model, 'last_mod_cos', None)
                hc = getattr(model, 'last_head_cos', None)
                pc = getattr(model, 'last_pv_cos', None)
                mcz = getattr(model, 'last_mod_cos_z', None)
                hcz = getattr(model, 'last_head_cos_z', None)
                pcz = getattr(model, 'last_pv_cos_z', None)
                st = getattr(model, 'last_align_state', None)
                mg = getattr(model, 'last_mod_gate', None)
                hg = getattr(model, 'last_head_gate', None)
                pg = getattr(model, 'last_pv_gate', None)

                bsz = len(mod_pred)
                all_mod_cos.append(mc.detach().float().cpu().numpy().reshape(-1) if mc is not None else np.zeros(bsz))
                all_head_cos.append(hc.detach().float().cpu().numpy().reshape(-1) if hc is not None else np.zeros(bsz))
                all_pv_cos.append(pc.detach().float().cpu().numpy().reshape(-1) if pc is not None else np.zeros(bsz))
                all_mod_cos_z.append(mcz.detach().float().cpu().numpy().reshape(-1) if mcz is not None else np.zeros(bsz))
                all_head_cos_z.append(hcz.detach().float().cpu().numpy().reshape(-1) if hcz is not None else np.zeros(bsz))
                all_pv_cos_z.append(pcz.detach().float().cpu().numpy().reshape(-1) if pcz is not None else np.zeros(bsz))
                all_align_state.append(st.detach().cpu().numpy().reshape(-1) if st is not None else np.full(bsz, -1, dtype=int))
                if mg is not None and hg is not None and pg is not None:
                    has_gate = True
                    all_mod_gate.append(mg.detach().float().cpu().numpy().reshape(-1))
                    all_head_gate.append(hg.detach().float().cpu().numpy().reshape(-1))
                    all_pv_gate.append(pg.detach().float().cpu().numpy().reshape(-1))

            lab = batch.get('has_label')
            if lab is not None:
                has_mask = True
                masks.append(lab.cpu().numpy().astype(bool).reshape(-1))
            else:
                masks.append(np.zeros(int(mod_pred.numel()), dtype=bool))

            # Xử lý an toàn nếu batch không chứa ground truth targets (ví dụ tập Test)
            if 'mod_avg' in batch and 'head_avg' in batch:
                all_mod_y.append(batch['mod_avg'].cpu().numpy().reshape(-1))
                all_head_y.append(batch['head_avg'].cpu().numpy().reshape(-1))
            else:
                all_mod_y.append(np.zeros(len(mod_pred)))
                all_head_y.append(np.zeros(len(head_pred)))

    mod = np.concatenate(all_mod) if all_mod else np.array([])
    head = np.concatenate(all_head) if all_head else np.array([])
    pv = np.concatenate(all_pv) if all_pv else np.array([])
    mod_y = np.concatenate(all_mod_y) if all_mod_y else np.array([])
    head_y = np.concatenate(all_head_y) if all_head_y else np.array([])
    mask = np.concatenate(masks) if has_mask else np.zeros(len(mod), dtype=bool)

    if return_diagnostics:
        diags = {
            'mod_cos': np.concatenate(all_mod_cos) if all_mod_cos else np.array([]),
            'head_cos': np.concatenate(all_head_cos) if all_head_cos else np.array([]),
            'pv_cos': np.concatenate(all_pv_cos) if all_pv_cos else np.array([]),
            'mod_cos_z': np.concatenate(all_mod_cos_z) if all_mod_cos_z else np.array([]),
            'head_cos_z': np.concatenate(all_head_cos_z) if all_head_cos_z else np.array([]),
            'pv_cos_z': np.concatenate(all_pv_cos_z) if all_pv_cos_z else np.array([]),
            'align_state': np.concatenate(all_align_state) if all_align_state else np.array([]),
            'mod_gate': np.concatenate(all_mod_gate) if has_gate and all_mod_gate else None,
            'head_gate': np.concatenate(all_head_gate) if has_gate and all_head_gate else None,
            'pv_gate': np.concatenate(all_pv_gate) if has_gate and all_pv_gate else None,
        }
        if return_all:
            if return_pv:
                return mod, head, pv, mod_y, head_y, mask, diags
            return mod, head, mod_y, head_y, mask, diags
        if has_mask and mask.any():
            return mod[mask], head[mask], mod_y[mask], head_y[mask], diags
        return mod, head, diags

    if return_all:
        if return_pv:
            return mod, head, pv, mod_y, head_y, mask
        return mod, head, mod_y, head_y, mask

    if has_mask and mask.any():
        return mod[mask], head[mask], mod_y[mask], head_y[mask]
    return mod, head

def unfreeze_top_layers(model, from_layer: int) -> None:
    """Unfreeze top encoder layers (from_layer..last) + final norm (if any)."""
    base = model.lm
    for attr in ('model', 'bert', 'base_model', 'transformer'):
        if hasattr(base, attr):
            base = getattr(base, attr)
            break

    layers = None
    if hasattr(base, 'layers'):
        layers = base.layers
    elif hasattr(base, 'encoder'):
        enc = base.encoder
        if hasattr(enc, 'layers'):
            layers = enc.layers
        elif hasattr(enc, 'layer'):
            layers = enc.layer
    if layers is None:
        raise AttributeError('Cannot locate the transformer encoder layers')

    for idx, layer in enumerate(layers):
        if idx >= from_layer:
            for name, param in layer.named_parameters():
                if '.linear.' in name:
                    continue   # base weight of a LoRAAdapter stays frozen
                param.requires_grad = True

    norm = getattr(base, 'final_norm', None)
    if norm is None and hasattr(base, 'encoder'):
        norm = getattr(base.encoder, 'final_norm', None)
    if norm is not None:
        for param in norm.parameters():
            param.requires_grad = True

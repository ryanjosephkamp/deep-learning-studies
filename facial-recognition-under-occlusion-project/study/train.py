"""
train.py - train one run of the project's model on identities, resumably.

The training crops sit on the device as one uint8 tensor. Each step draws a batch, flips it
horizontally at random, and, for runs with occlusion augmentation, draws the project's centered
black square on each image with probability 0.5 (`occlusion_probability`), with coverage uniform
in the configured range (the original recipe's 10 to 50%), so the evaluation's bands, off-center squares, and gray or noise
fills are never seen in training. A checkpoint (weights, head, optimizer, schedule, and random
states) is saved every few minutes and at the end of each epoch; a restarted run resumes from it.
"""

import os
import time
import math

import numpy as np
import pandas as pd
import torch

from common import atomic_path, commit, seed_everything
from data import torch_center_square
from nets import build_model, CosFaceHead, batch_hard_triplet


def _lr_lambda(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def _augment(x_uint8_bgr, run, tc, gen):
    """uint8 BGR (B, H, W, 3) on device -> float RGB (B, 3, H, W) in [-1, 1], flipped and occluded."""
    x = x_uint8_bgr.flip(-1).permute(0, 3, 1, 2).float()  # reverse the channel axis: BGR -> RGB
    flip = torch.rand(len(x), device=x.device, generator=gen) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    if run['augment']:
        lo, hi = tc['occlusion_range']
        cov = lo + (hi - lo) * torch.rand(len(x), device=x.device, generator=gen)
        apply = torch.rand(len(x), device=x.device, generator=gen) < tc['occlusion_probability']
        x = torch_center_square(x, cov, apply)
    return x.div_(127.5).sub_(1.0)


def _triplet_batches(labels_np, steps, p, k, rng):
    """P identities x K images each (with replacement when an identity has fewer than K)."""
    by_id = {}
    for i, l in enumerate(labels_np):
        by_id.setdefault(int(l), []).append(i)
    ids = np.array(sorted(by_id))
    for _ in range(steps):
        chosen = rng.choice(ids, size=p, replace=False)
        yield np.concatenate([rng.choice(by_id[c], size=k, replace=len(by_id[c]) < k) for c in chosen])


def train_run(run, seed, crops, labels, cfg, device, out_dir, log=print):
    """Train one (run, seed) and write out_dir/model.pt. crops: uint8 BGR tensor (N, H, W, 3) on device."""
    tc = cfg['train']
    final = os.path.join(out_dir, 'model.pt')
    ckpt_path = os.path.join(out_dir, 'checkpoint.pt')
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(seed)
    n_classes = int(labels.max().item()) + 1
    model = build_model(run['attention'], tc['embedding_size']).to(device)
    head = CosFaceHead(n_classes, tc['embedding_size'], tc['cosface_scale'], tc['cosface_margin']).to(device) \
        if run['loss'] == 'cosface' else None
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=tc['lr'], weight_decay=tc['weight_decay'])

    n = len(crops)
    if run['loss'] == 'cosface':
        steps_per_epoch = max(1, n // tc['batch_size'])
    else:
        steps_per_epoch = max(1, n // (tc['triplet_identities_per_batch'] * tc['triplet_images_per_identity']))
    total = steps_per_epoch * tc['epochs']
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(total, steps_per_epoch * tc['warmup_epochs']))
    gen = torch.Generator(device=device).manual_seed(seed)
    start_epoch, history = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        if head is not None:
            head.load_state_dict(ck['head'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        gen.set_state(ck['gen'].cpu())
        torch.set_rng_state(ck['torch_rng'])
        start_epoch, history = ck['epoch'], ck['history']
        log(f'  resumed {os.path.basename(out_dir)} at epoch {start_epoch}')

    labels_np = labels.cpu().numpy()
    last_ckpt = time.time()
    use_amp = device.type == 'cuda'
    for epoch in range(start_epoch, tc['epochs']):
        model.train()
        t0, total_loss = time.time(), 0.0
        if run['loss'] == 'cosface':
            order = torch.randperm(n, device=device, generator=gen)
            batches = (order[i * tc['batch_size']:(i + 1) * tc['batch_size']] for i in range(steps_per_epoch))
        else:
            rng = np.random.default_rng([seed, epoch])
            batches = (torch.from_numpy(b).to(device) for b in _triplet_batches(
                labels_np, steps_per_epoch, tc['triplet_identities_per_batch'],
                tc['triplet_images_per_identity'], rng))
        for idx in batches:
            x = _augment(crops[idx], run, tc, gen)
            y = labels[idx]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                emb = model(x)
            emb = emb.float()
            loss = head(emb, y) if head is not None else batch_hard_triplet(emb, y, tc['triplet_margin'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
        history.append({'epoch': epoch + 1, 'loss': total_loss / steps_per_epoch,
                        'lr': sched.get_last_lr()[0], 'seconds': time.time() - t0})
        done = epoch + 1 == tc['epochs']
        if done or time.time() - last_ckpt > 60 * tc['checkpoint_minutes']:
            tmp = atomic_path(ckpt_path)
            torch.save({'model': model.state_dict(), 'head': head.state_dict() if head is not None else None,
                        'opt': opt.state_dict(), 'sched': sched.state_dict(), 'gen': gen.get_state(),
                        'torch_rng': torch.get_rng_state(), 'epoch': epoch + 1, 'history': history}, tmp)
            commit(tmp, ckpt_path)
            last_ckpt = time.time()
    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'train_log.csv'), index=False)
    tmp = atomic_path(final)
    torch.save({'model': model.state_dict(), 'run': run, 'seed': seed, 'n_classes': n_classes,
                'n_images': n, 'epochs': tc['epochs']}, tmp)
    commit(tmp, final)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

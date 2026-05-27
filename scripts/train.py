"""
Training script for STGIN
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import get_config
from models.stgin import STGIN
from data.tbd_dataset import build_dataloader, collect_all_captions
from utils.vocab import Vocabulary, build_concept_vocab
from utils.metrics import evaluate_captions


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_vocab_if_needed(cfg, save_dir):
    vocab_path   = os.path.join(save_dir, f'vocab_{cfg.dataset_name}.json')
    concept_path = os.path.join(save_dir, f'concepts_{cfg.dataset_name}.json')

    if os.path.exists(vocab_path) and os.path.exists(concept_path):
        vocab = Vocabulary.load(vocab_path)
        with open(concept_path) as f:
            concept_vocab = json.load(f)
        print(f"[vocab] 加载: |V|={len(vocab)}, |concepts|={len(concept_vocab)}")
        return vocab, concept_vocab

    print("[vocab] 从标注文件构建词表...")
    # ── 直接从你的标注格式读取所有 captions ──
    all_caps = collect_all_captions(cfg.annotation_file)
    print(f"[vocab] 共读取到 {len(all_caps)} 句 caption")

    vocab = Vocabulary(min_freq=cfg.min_word_freq, max_size=cfg.vocab_size)
    vocab.build(all_caps)

    os.makedirs(save_dir, exist_ok=True)
    vocab.save(vocab_path)

    concept_vocab = build_concept_vocab(
        all_caps, num_concepts=cfg.num_concepts, pos_filter=False)
    with open(concept_path, 'w', encoding='utf-8') as f:
        json.dump(concept_vocab, f, indent=2, ensure_ascii=False)

    print(f"[vocab] 构建完成: |V|={len(vocab)}, |concepts|={len(concept_vocab)}")
    return vocab, concept_vocab


def move_to_device(batch, device):
    return {k: v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, vocab, cfg, device, show_samples=3):
    model.eval()
    hyps, refs = {}, {}
    for batch in loader:
        batch = move_to_device(batch, device)
        seqs = model.generate(batch, mode=cfg.decode_mode,
                              beam_size=cfg.beam_size)
        for vid, seq, gt in zip(batch['video_ids'], seqs, batch['captions']):
            hyps[vid] = vocab.decode(seq.cpu().tolist())
            refs[vid] = gt

    # 打印几条生成样例，直观观察模型输出
    print("  [样例对比]")
    for i, (vid, hyp) in enumerate(list(hyps.items())[:show_samples]):
        ref_str = refs[vid][0] if refs[vid] else ""
        print(f"  [{i+1}] 生成: {''.join(hyp.split())}")
        print(f"       参考: {ref_str[:50]}{'...' if len(ref_str)>50 else ''}")
    return evaluate_captions(hyps, refs), hyps


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, epoch, writer):
    model.train()
    total = {'loss': 0., 'loss_cap': 0., 'loss_scd': 0.}
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        batch['lambda_scd'] = cfg.lambda_scd
        optimizer.zero_grad()
        out = model(batch, return_loss=True)
        if torch.isnan(out['loss']):
            print(f"[epoch {epoch} step {step}] NaN loss, skip"); continue
        out['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        for k in total: total[k] += out[k].item() if k in out else out['loss'].item()
        n += 1
        if (step + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"[epoch {epoch} {step+1}/{len(loader)}] "
                  f"loss={out['loss'].item():.4f} "
                  f"cap={out['loss_cap'].item():.4f} "
                  f"scd={out['loss_scd'].item():.4f} "
                  f"lr={cur_lr:.2e}  ({elapsed:.1f}s)")
            if writer:
                gs = (epoch-1)*len(loader)+step
                writer.add_scalar('train/loss',     out['loss'].item(),     gs)
                writer.add_scalar('train/loss_cap', out['loss_cap'].item(), gs)
                writer.add_scalar('train/loss_scd', out['loss_scd'].item(), gs)
            t0 = time.time()
    return {k: v/max(n,1) for k, v in total.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',          type=str, default='TBD')
    parser.add_argument('--annotation_file',  type=str, default=None)
    parser.add_argument('--feature_dir',      type=str, default=None)
    parser.add_argument('--save_dir',         type=str, default=None)
    parser.add_argument('--num_epochs',       type=int, default=None)
    parser.add_argument('--batch_size',       type=int, default=None)
    parser.add_argument('--lr',               type=float, default=None)
    parser.add_argument('--train_ratio',      type=float, default=0.6)
    parser.add_argument('--val_ratio',        type=float, default=0.2)
    parser.add_argument('--resume',           type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.dataset)
    if args.annotation_file: cfg.annotation_file = args.annotation_file
    if args.feature_dir:     cfg.feature_dir     = args.feature_dir
    if args.save_dir:        cfg.save_dir        = args.save_dir
    if args.num_epochs:      cfg.num_epochs      = args.num_epochs
    if args.batch_size:      cfg.batch_size      = args.batch_size
    if args.lr:              cfg.learning_rate   = args.lr

    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    vocab, concept_vocab = build_vocab_if_needed(cfg, cfg.save_dir)

    # ── DataLoader（传入 train_ratio/val_ratio 供随机划分） ──
    common = dict(
        max_caption_len=cfg.max_caption_len,
        num_frames=cfg.num_frames,
        max_boxes=cfg.max_boxes,
        obj_feat_dim=cfg.obj_feat_dim,
        appear_dim=cfg.appear_dim,
        motion_dim=cfg.motion_dim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=cfg.seed,
    )
    train_loader = build_dataloader(
        cfg.annotation_file, cfg.feature_dir, vocab, concept_vocab,
        split='train', batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, **common)
    val_loader = build_dataloader(
        cfg.annotation_file, cfg.feature_dir, vocab, concept_vocab,
        split='val', batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, **common)

    # 用实际构建出的概念数，而不是配置里的固定值
    actual_num_concepts = len(concept_vocab)
    print(f"[model] 实际 num_concepts={actual_num_concepts}（配置值={cfg.num_concepts}）")

    model = STGIN(
        vocab_size=len(vocab),
        num_concepts=actual_num_concepts,
        obj_feat_dim=cfg.obj_feat_dim,
        appear_dim=cfg.appear_dim,
        motion_dim=cfg.motion_dim,
        visual_out_dim=cfg.visual_out_dim,
        concept_embed_dim=cfg.concept_embed_dim,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_caption_len,
        pad_id=vocab.pad_id, bos_id=vocab.bos_id, eos_id=vocab.eos_id,
        num_gcn_layers=cfg.num_gcn_layers,
        dist_threshold=cfg.dist_threshold,
        beta=cfg.beta_loss,
        label_smoothing=getattr(cfg, 'label_smoothing', 0.0),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] STGIN 参数量: {n_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # Warmup + Cosine 学习率调度
    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = getattr(cfg, 'warmup_steps', 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch, best_score = 1, -1.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch  = ckpt['epoch'] + 1
        best_score   = ckpt.get('best_score', -1.0)
        print(f"[resume] epoch={ckpt['epoch']}  best={best_score:.4f}")

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.num_epochs} =====")
        stats = train_one_epoch(model, train_loader, optimizer, scheduler,
                                cfg, device, epoch, writer)
        print(f"[epoch {epoch}] avg: {stats}")

        scores, hyps = evaluate(model, val_loader, vocab, cfg, device)
        print(f"[epoch {epoch}] val:", {k: f"{v:.2f}" for k, v in scores.items()})
        for k, v in scores.items():
            writer.add_scalar(f'val/{k}', v, epoch)

        ckpt = {'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score, 'scores': scores}
        torch.save(ckpt, os.path.join(cfg.save_dir, 'last.pt'))

        cur = scores.get(cfg.best_metric, 0.0)
        if cur > best_score:
            best_score = cur
            ckpt['best_score'] = best_score
            torch.save(ckpt, os.path.join(cfg.save_dir, 'best.pt'))
            with open(os.path.join(cfg.save_dir, 'best_preds.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(hyps, f, indent=2, ensure_ascii=False)
            print(f"  ★ 新最优 {cfg.best_metric}={best_score:.4f}")

    writer.close()
    print(f"\n训练结束。最优 {cfg.best_metric} = {best_score:.4f}")


if __name__ == "__main__":
    main()
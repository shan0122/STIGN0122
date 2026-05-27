"""
Evaluation / Inference script for STGIN
对应论文 Section 4.3:
  - 在 test 集上做 beam search (beam_size=6) 解码
  - 输出 BLEU_1/2/3/4, METEOR, ROUGE_L, CIDEr
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import get_config
from models.stgin import STGIN
from data.tbd_dataset import build_dataloader
from utils.vocab import Vocabulary
from utils.metrics import evaluate_captions


def move_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def run_inference(model, loader, vocab, cfg, device):
    model.eval()
    hyps, refs = {}, {}
    for i, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        sequences = model.generate(batch, mode=cfg.decode_mode,
                                   beam_size=cfg.beam_size)
        for vid, seq, gt_caps in zip(batch['video_ids'], sequences, batch['captions']):
            sent = vocab.decode(seq.cpu().tolist())
            hyps[vid] = sent
            refs[vid] = gt_caps
        if (i + 1) % 10 == 0:
            print(f"  [eval] batch {i+1}/{len(loader)}")
    return hyps, refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TBD')
    parser.add_argument('--ckpt',    type=str, required=True)
    parser.add_argument('--split',   type=str, default='test',
                        choices=['val', 'test'])
    parser.add_argument('--annotation_file', type=str, default=None)
    parser.add_argument('--feature_dir',     type=str, default=None)
    parser.add_argument('--output',          type=str, default='predictions.json')
    parser.add_argument('--beam_size',       type=int, default=None)
    args = parser.parse_args()

    cfg = get_config(args.dataset)
    if args.annotation_file: cfg.annotation_file = args.annotation_file
    if args.feature_dir:     cfg.feature_dir     = args.feature_dir
    if args.beam_size:       cfg.beam_size       = args.beam_size

    device = cfg.device if torch.cuda.is_available() else 'cpu'

    # 词表 / 概念表
    vocab_path = os.path.join(cfg.save_dir, f'vocab_{cfg.dataset_name}.json')
    concept_path = os.path.join(cfg.save_dir, f'concepts_{cfg.dataset_name}.json')
    vocab = Vocabulary.load(vocab_path)
    with open(concept_path) as f:
        concept_vocab = json.load(f)

    # DataLoader
    loader = build_dataloader(
        cfg.annotation_file, cfg.feature_dir, vocab, concept_vocab,
        split=args.split, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_caption_len=cfg.max_caption_len,
        num_frames=cfg.num_frames, max_boxes=cfg.max_boxes,
        obj_feat_dim=cfg.obj_feat_dim, appear_dim=cfg.appear_dim,
        motion_dim=cfg.motion_dim,
    )

    # 模型
    actual_num_concepts = len(concept_vocab)
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
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        num_gcn_layers=cfg.num_gcn_layers,
        dist_threshold=cfg.dist_threshold,
        beta=cfg.beta_loss,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    print(f"[ckpt] loaded epoch={ckpt.get('epoch','?')} "
          f"best_score={ckpt.get('best_score','?')}")

    # 推理
    print(f"[eval] beam_size={cfg.beam_size}, split={args.split}")
    hyps, refs = run_inference(model, loader, vocab, cfg, device)

    # 评估
    scores = evaluate_captions(hyps, refs)
    print("\n========== Results ==========")
    for k, v in scores.items():
        print(f"  {k:10s}: {v:.4f}")

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'scores': scores,
            'predictions': hyps,
            'references': refs,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
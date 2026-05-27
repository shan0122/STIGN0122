"""
Step 1: 预提取视频特征并缓存为 .pt

用法:
  python scripts/precompute_features.py \
      --dataset TBD \
      --video_dir  data/videos/tbd \
      --output_dir data/features/tbd \
      --annotation_file data/annotations/tbd_annotations.json

特征保存规则（与标注文件的 video_path 字段一一对应）:
  video_path: "clips_class_02/0001.mp4"
  →  特征文件: data/features/tbd/clips_class_02/0001.pt
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import get_config
from data.tbd_dataset import load_annotations, precompute_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',          type=str, default='TBD')
    parser.add_argument('--video_dir',        type=str, default=None,
                        help='视频根目录，如 data/videos/tbd')
    parser.add_argument('--output_dir',       type=str, default=None,
                        help='特征输出目录，如 data/features/tbd')
    parser.add_argument('--annotation_file',  type=str, default=None,
                        help='标注 JSON 文件路径')
    parser.add_argument('--c3d_weights',      type=str, default=None)
    parser.add_argument('--yolo_weights',     type=str, default=None,
                        help='本地 YOLOv5 权重路径，如 weights/yolov5s.pt（离线时使用）')
    parser.add_argument('--ext',              type=str, default='.mp4')
    parser.add_argument('--device',           type=str, default=None)
    args = parser.parse_args()

    cfg        = get_config(args.dataset)
    video_dir  = args.video_dir       or cfg.video_dir
    output_dir = args.output_dir      or cfg.feature_dir
    anno_path  = args.annotation_file or cfg.annotation_file
    device     = args.device          or cfg.device

    # 读标注，收集 video_id 列表 和 video_id → video_path 映射
    records    = load_annotations(anno_path)
    # 去重（同一视频可能有多条 caption）
    seen = {}
    for r in records:
        vid = r['video_id']
        if vid not in seen:
            seen[vid] = r['video_path']

    video_ids   = list(seen.keys())
    video_paths = seen          # {video_id: 'clips_class_02/0001.mp4'}

    print(f"[precompute] 共 {len(video_ids)} 个唯一视频")
    print(f"  视频目录 : {video_dir}")
    print(f"  特征输出 : {output_dir}")
    print(f"  设备     : {device}")
    print(f"\n前5条 video_path 示例:")
    for vid in video_ids[:5]:
        print(f"  {vid}  →  {video_paths[vid]}")
    print()

    precompute_features(
        video_dir=video_dir,
        output_dir=output_dir,
        video_ids=video_ids,
        video_paths=video_paths,
        device=device,
        num_frames=cfg.num_frames,
        max_boxes=cfg.max_boxes,
        c3d_weights=args.c3d_weights,
        yolo_weights=args.yolo_weights,
        ext=args.ext,
    )


if __name__ == "__main__":
    main()
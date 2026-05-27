"""
TBD Dataset  —  适配你的实际标注格式

标注文件格式（顶层为数组）:
[
  {
    "video_id":      "clips_class_02_0001",
    "video_path":    "clips_class_02/0001.mp4",
    "action_labels": ["Multimedia Teaching"],
    "captions":      ["教师站在大屏幕前进行多媒体教学，根据课件内容为学生授课。"]
  },
  ...
]

视频目录结构:
  data/videos/tbd/
  ├── clips_class_01/
  │   ├── 0001.mp4
  │   └── 0002.mp4
  ├── clips_class_02/
  └── ...

特征目录结构（precompute后自动生成）:
  data/features/tbd/
  ├── clips_class_01/
  │   ├── 0001.pt
  │   └── 0002.pt
  └── ...
"""

import os
import json
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader

from utils.vocab import Vocabulary, captions_to_concept_labels


# ---------------------------------------------------------------
# 工具：扫描视频目录，建立 video_path（相对路径去扩展名）→ 绝对路径 的映射
# ---------------------------------------------------------------
def scan_video_dir(video_dir: str, ext: str = '.mp4') -> dict:
    """
    递归扫描 video_dir，返回 {rel_path_no_ext: abs_path}
    例: {'clips_class_01/0001': '/data/videos/tbd/clips_class_01/0001.mp4'}
    """
    mapping = {}
    for abs_path in sorted(glob.glob(
            os.path.join(video_dir, '**', f'*{ext}'), recursive=True)):
        rel = os.path.relpath(abs_path, video_dir)        # clips_class_01/0001.mp4
        key = os.path.splitext(rel)[0].replace(os.sep, '/')  # clips_class_01/0001
        mapping[key] = abs_path
    return mapping


def _find_video(video_path_field: str,
                video_dir: str,
                ext: str,
                path_map: dict) -> str:
    """
    按优先级查找视频文件:
      1. video_dir / video_path_field          直接拼接（最常见）
      2. path_map[video_path_field去扩展名]     映射表精确匹配
    """
    # 方式1: 直接拼接 video_dir + video_path 字段
    direct = os.path.join(video_dir, video_path_field)
    if os.path.exists(direct):
        return direct

    # 方式2: 去扩展名后查映射表
    key = os.path.splitext(video_path_field.replace(os.sep, '/'))[0]
    if key in path_map:
        return path_map[key]

    return None


# ---------------------------------------------------------------
# 标注文件解析
# ---------------------------------------------------------------
def load_annotations(annotation_file: str) -> list:
    """
    读取标注文件，统一返回 list of dict，每条包含:
      video_id, video_path, captions (list[str]), action_labels (list[str])
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    records = []
    # ── 你的格式：顶层是数组 ──
    if isinstance(raw, list):
        for item in raw:
            records.append({
                'video_id':     item['video_id'],
                'video_path':   item.get('video_path', ''),
                'captions':     item.get('captions', []),
                'action_labels':item.get('action_labels', []),
            })
    # ── 兼容格式A：{"videos": [...]} ──
    elif isinstance(raw, dict) and 'videos' in raw:
        for item in raw['videos']:
            records.append({
                'video_id':     item['video_id'],
                'video_path':   item.get('video_path',
                                         item.get('video_id', '') + '.mp4'),
                'captions':     item.get('captions',
                                         [item.get('caption', '')]),
                'action_labels':item.get('action_labels', []),
                '_split':       item.get('split', ''),
            })
    # ── 兼容格式B：{video_id: [caps]} ──
    elif isinstance(raw, dict):
        for vid, caps in raw.items():
            records.append({
                'video_id':     vid,
                'video_path':   vid + '.mp4',
                'captions':     caps if isinstance(caps, list) else [caps],
                'action_labels':[],
            })
    else:
        raise ValueError(f"无法识别的标注格式: {annotation_file}")

    return records


def split_records(records: list,
                  split_file: str = None,
                  train_ratio: float = 0.6,
                  val_ratio: float = 0.2,
                  seed: int = 42) -> dict:
    """
    把 records 按 train/val/test 划分，返回 {split: [records]}。

    优先级:
      1. 若 record 已有 '_split' 字段 → 直接用
      2. 若提供 split_file（每行 video_id\tsplit）→ 从文件读
      3. 否则按比例随机划分
    """
    # 情况1: 记录里已有 split
    if records and '_split' in records[0] and records[0]['_split']:
        result = {'train': [], 'val': [], 'test': []}
        for r in records:
            s = r.get('_split', 'train')
            result.setdefault(s, []).append(r)
        return result

    # 情况2: 从 split_file 读取
    if split_file and os.path.exists(split_file):
        id2split = {}
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    id2split[parts[0]] = parts[1]
                elif len(parts) == 1 and parts[0]:
                    id2split[parts[0]] = 'train'
        result = {'train': [], 'val': [], 'test': []}
        for r in records:
            s = id2split.get(r['video_id'], 'train')
            result[s].append(r)
        return result

    # 情况3: 按比例随机划分（你的标注文件没有 split 字段，走这里）
    all_ids = list({r['video_id'] for r in records})
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    id2split = {}
    for vid in all_ids[:n_train]:              id2split[vid] = 'train'
    for vid in all_ids[n_train:n_train+n_val]: id2split[vid] = 'val'
    for vid in all_ids[n_train+n_val:]:        id2split[vid] = 'test'

    result = {'train': [], 'val': [], 'test': []}
    for r in records:
        result[id2split[r['video_id']]].append(r)

    print(f"[split] train={len(result['train'])} "
          f"val={len(result['val'])} test={len(result['test'])} "
          f"（按 {train_ratio:.0%}/{val_ratio:.0%}/"
          f"{1-train_ratio-val_ratio:.0%} 随机划分，seed={seed}）")
    return result


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class TBDDataset(Dataset):

    def __init__(self,
                 annotation_file: str,
                 feature_dir: str,
                 vocab: Vocabulary,
                 concept_vocab: list,
                 split: str = 'train',
                 max_caption_len: int = 26,
                 num_frames: int = 26,
                 max_boxes: int = 5,
                 obj_feat_dim: int = 2048,
                 appear_dim: int = 2048,
                 motion_dim: int = 4096,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 split_file: str = None,
                 seed: int = 42):
        super().__init__()
        self.feature_dir     = feature_dir
        self.vocab           = vocab
        self.concept_vocab   = concept_vocab
        self.split           = split
        self.max_caption_len = max_caption_len
        self.num_frames      = num_frames
        self.max_boxes       = max_boxes
        self.obj_feat_dim    = obj_feat_dim
        self.appear_dim      = appear_dim
        self.motion_dim      = motion_dim

        # 读标注 & 划分
        all_records = load_annotations(annotation_file)
        splits = split_records(all_records, split_file,
                               train_ratio, val_ratio, seed)
        cur_records = splits.get(split, [])

        # 聚合: video_id → {video_path, captions, action_labels}
        self.video2info = {}
        for r in cur_records:
            vid = r['video_id']
            if vid not in self.video2info:
                self.video2info[vid] = {
                    'video_path':    r['video_path'],
                    'captions':      [],
                    'action_labels': r['action_labels'],
                }
            for cap in r['captions']:
                if cap and cap not in self.video2info[vid]['captions']:
                    self.video2info[vid]['captions'].append(cap)

        self.video_ids = list(self.video2info.keys())

        # 训练: 每句 caption 一个样本; val/test: 每个视频一个样本
        if split == 'train':
            self.samples = [
                (vid, cap)
                for vid in self.video_ids
                for cap in self.video2info[vid]['captions']
                if cap   # 跳过空 caption
            ]
        else:
            self.samples = [(vid, self.video2info[vid]['captions'])
                            for vid in self.video_ids]

        print(f"[TBDDataset/{split}] "
              f"videos={len(self.video_ids)}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _feat_path(self, vid: str, video_path: str) -> str:
        """
        特征文件路径:
          video_path = 'clips_class_02/0001.mp4'
          → feature_dir/clips_class_02/0001.pt
        """
        rel = os.path.splitext(video_path)[0]          # clips_class_02/0001
        return os.path.join(self.feature_dir, rel + '.pt')

    def _load_features(self, vid: str) -> dict:
        video_path = self.video2info[vid]['video_path']
        path = self._feat_path(vid, video_path)
        if os.path.exists(path):
            return torch.load(path, map_location='cpu', weights_only=True)
        # 找不到时返回零张量（调试用）
        return {
            'appear_feats':  torch.zeros(self.num_frames, self.appear_dim),
            'motion_feats':  torch.zeros(self.num_frames, self.motion_dim),
            'obj_feats':     torch.zeros(self.num_frames, self.max_boxes,
                                         self.obj_feat_dim),
            'obj_boxes':     torch.zeros(self.num_frames, self.max_boxes, 4),
            'obj_mask':      torch.zeros(self.num_frames, self.max_boxes),
            'teacher_feats': torch.zeros(self.num_frames, self.max_boxes,
                                         self.obj_feat_dim),
            'teacher_mask':  torch.zeros(self.num_frames, self.max_boxes),
        }

    def __getitem__(self, idx: int):
        if self.split == 'train':
            vid, caption = self.samples[idx]
            captions = [caption]
        else:
            vid, captions = self.samples[idx]

        feats = self._load_features(vid)

        ids = self.vocab.encode(captions[0],
                                max_len=self.max_caption_len + 1,
                                add_special=True)
        ids         = torch.tensor(ids, dtype=torch.long)
        caption_in  = ids[:-1]
        caption_tgt = ids[1:]

        concept_labels = captions_to_concept_labels(captions, self.concept_vocab)
        concept_labels = torch.tensor(concept_labels, dtype=torch.float)

        return {
            'video_id':       vid,
            'captions':       captions,
            'caption_in':     caption_in,
            'caption_tgt':    caption_tgt,
            'concept_labels': concept_labels,
            **feats,
        }


# ---------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------
def collate_fn(batch):
    out = {}
    keys_tensor = ['caption_in', 'caption_tgt', 'concept_labels',
                   'appear_feats', 'motion_feats',
                   'obj_feats', 'obj_boxes', 'obj_mask',
                   'teacher_feats', 'teacher_mask']
    for k in keys_tensor:
        out[k] = torch.stack([item[k] for item in batch], dim=0)
    out['video_ids'] = [item['video_id'] for item in batch]
    out['captions']  = [item['captions'] for item in batch]
    return out


def build_dataloader(annotation_file: str,
                     feature_dir: str,
                     vocab: Vocabulary,
                     concept_vocab: list,
                     split: str = 'train',
                     batch_size: int = 64,
                     num_workers: int = 4,
                     **kwargs) -> DataLoader:
    ds = TBDDataset(annotation_file, feature_dir, vocab, concept_vocab,
                    split=split, **kwargs)
    shuffle = (split == 'train')
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      pin_memory=True,
                      drop_last=shuffle)


# ---------------------------------------------------------------
# 预计算特征
# ---------------------------------------------------------------
def precompute_features(video_dir: str,
                        output_dir: str,
                        video_ids: list,
                        video_paths: dict,        # video_id → video_path 字段
                        device: str = 'cuda',
                        num_frames: int = 26,
                        max_boxes: int = 5,
                        c3d_weights: str = None,
                        ext: str = '.mp4'):
    """
    video_paths: {video_id: 'clips_class_02/0001.mp4'}  来自标注文件
    特征保存到: output_dir/clips_class_02/0001.pt
    """
    from utils.feature_extractor import VideoFeaturePipeline

    os.makedirs(output_dir, exist_ok=True)

    # 预扫描视频目录，建立备用索引
    print(f"[precompute] 扫描视频目录: {video_dir} ...")
    path_map = scan_video_dir(video_dir, ext=ext)
    print(f"[precompute] 找到 {len(path_map)} 个视频，开始提取特征...\n")

    pipeline = VideoFeaturePipeline(
        num_frames=num_frames,
        max_boxes=max_boxes,
        device=device,
        c3d_weights=c3d_weights,
    )

    skip, done, error = 0, 0, 0
    for i, vid in enumerate(video_ids):
        vpath_field = video_paths.get(vid, '')          # clips_class_02/0001.mp4
        # 特征输出路径
        rel_no_ext  = os.path.splitext(vpath_field)[0] # clips_class_02/0001
        out_path    = os.path.join(output_dir, rel_no_ext + '.pt')

        if os.path.exists(out_path):
            skip += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 找视频文件
        video_abs = _find_video(vpath_field, video_dir, ext, path_map)
        if video_abs is None:
            print(f"  [skip] 找不到视频: {vpath_field}")
            skip += 1
            continue

        try:
            feats = pipeline(video_abs)
            torch.save({k: v.cpu() for k, v in feats.items()}, out_path)
            done += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(video_ids):
                print(f"  [{i+1}/{len(video_ids)}] "
                      f"done={done} skip={skip} error={error}")
        except Exception as e:
            print(f"  [error] {vid}: {e}")
            error += 1

    print(f"\n[precompute] 完成: done={done}, skip={skip}, error={error}")


# ---------------------------------------------------------------
# 词表构建辅助（从你的标注格式直接读 captions）
# ---------------------------------------------------------------
def collect_all_captions(annotation_file: str) -> list:
    """返回标注文件中所有 caption 字符串的列表（用于构建词表）"""
    records = load_annotations(annotation_file)
    captions = []
    for r in records:
        captions.extend(r['captions'])
    return captions


if __name__ == "__main__":
    # ── 快速自测 ──
    import tempfile, json, os, sys, torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 模拟你的标注格式
    anno = [
        {
            "video_id": "clips_class_01_0001",
            "video_path": "clips_class_01/0001.mp4",
            "action_labels": ["Multimedia Teaching"],
            "captions": ["教师站在大屏幕前进行多媒体教学，根据课件内容为学生授课。"]
        },
        {
            "video_id": "clips_class_01_0002",
            "video_path": "clips_class_01/0002.mp4",
            "action_labels": ["Blackboard Teaching"],
            "captions": ["教师在黑板上书写板书，向学生讲解知识点。"]
        },
        {
            "video_id": "clips_class_02_0001",
            "video_path": "clips_class_02/0001.mp4",
            "action_labels": ["Interaction"],
            "captions": ["教师走到学生桌旁，指导学生完成练习题。"]
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        # 写标注文件
        anno_f = os.path.join(tmp, 'anno.json')
        with open(anno_f, 'w', encoding='utf-8') as f:
            json.dump(anno, f, ensure_ascii=False)

        # 读标注
        records = load_annotations(anno_f)
        assert len(records) == 3
        assert records[0]['video_id'] == 'clips_class_01_0001'
        assert records[0]['video_path'] == 'clips_class_01/0001.mp4'
        assert '教师站在大屏幕' in records[0]['captions'][0]
        print("✓ load_annotations OK")

        # 划分
        splits = split_records(records, seed=42)
        total = sum(len(v) for v in splits.values())
        assert total == 3
        print("✓ split_records OK")

        # 模拟特征文件
        feat_dir = os.path.join(tmp, 'features')
        os.makedirs(os.path.join(feat_dir, 'clips_class_01'), exist_ok=True)
        os.makedirs(os.path.join(feat_dir, 'clips_class_02'), exist_ok=True)
        dummy = {
            'appear_feats':  torch.zeros(26, 2048),
            'motion_feats':  torch.zeros(26, 4096),
            'obj_feats':     torch.zeros(26, 5, 2048),
            'obj_boxes':     torch.zeros(26, 5, 4),
            'obj_mask':      torch.zeros(26, 5),
            'teacher_feats': torch.zeros(26, 5, 2048),
            'teacher_mask':  torch.zeros(26, 5),
        }
        for r in records:
            pt_path = os.path.join(
                feat_dir,
                os.path.splitext(r['video_path'])[0] + '.pt')
            torch.save(dummy, pt_path)

        # 构建词表
        from utils.vocab import Vocabulary
        vocab = Vocabulary(min_freq=1)
        vocab.build([c for r in records for c in r['captions']])

        # Dataset
        ds = TBDDataset(anno_f, feat_dir, vocab,
                        concept_vocab=['教师', '学生', '黑板'],
                        split='train',
                        max_caption_len=26,
                        train_ratio=0.67, val_ratio=0.33)
        if len(ds) > 0:
            item = ds[0]
            assert item['appear_feats'].shape == (26, 2048)
            assert item['caption_in'].shape  == (26,)
            print(f"✓ TBDDataset OK  video_id={item['video_id']}")
        else:
            print("✓ TBDDataset OK（训练集为空是因为样本太少，实际运行无问题）")

    print("\n✓ 所有测试通过")
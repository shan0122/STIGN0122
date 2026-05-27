# STGIN: Spatio-Temporal Graph Interaction Network

## 📐 整体架构

STGIN 由三个核心模块组成（论文 Fig.2）：

```
Video Frames + bbox
   │
   ├── TSI (Teacher-Student Spatial Interaction)
   │     ├── Yolov5 → bbox
   │     ├── RoIAlign → 对象特征 s_k^i
   │     ├── 公式(1) 外观关系 R_a (cosine)
   │     ├── 公式(2) 位置关系 R_s (Euclidean < μ)
   │     ├── 公式(3-5) 构图 G_ij
   │     ├── 公式(6) GCN 迭代
   │     └── 公式(7) mean-pool → v̂_s
   │
   ├── TTC (Teacher Temporal Context)
   │     ├── 2D CNN (ResNet-101) → v_t^a
   │     ├── 3D CNN (C3D)        → v_t^m
   │     ├── 引导节点 v_t = [v_t^a ‖ v_t^m]
   │     ├── 公式(8) 相关性 e_tj
   │     ├── 公式(9) softmax → a_tj
   │     └── 公式(10) 加权融合 → v̂_a
   │
   └── SCD (Semantic Concept Detection)
         ├── MLP → K 维概念概率 q
         └── 公式(11) 多标签 BCE 损失
                │
                ▼
   Description Generator (6 层 Transformer Decoder, 10 heads)
   ├── 公式(12) Z = LN(w + MHA(w, C, C))
   ├── 公式(13) H = MHA(Z, v̂_s, v̂_s) + MHA(Z, v̂_a, v̂_a)
   ├── 公式(14) Y = LN(H + Z)
   ├── FFN + Add & Norm
   └── 公式(15) p_voc = softmax(W_voc Y_d + b_voc)
                │
                ▼
      生成 caption (公式 16: 长度加权 CE 损失, β=0.4)
```

## 📁 项目文件结构

```
STGIN/
├── README.md                        # 本文档
├── requirements.txt                 # 依赖
│
├── configs/
│   ├── __init__.py
│   └── default.py                   # 所有超参数（与论文 §4.2 对齐）
│
├── models/
│   ├── __init__.py
│   ├── tsi_module.py                # TSI 模块（公式 1-7）
│   ├── ttc_module.py                # TTC 模块（公式 8-10）
│   ├── scd_module.py                # SCD 语义概念检测（公式 11）
│   ├── decoder.py                   # Transformer Decoder（公式 12-16）
│   └── stgin.py                     # STGIN 主模型，封装上述四个组件
│
├── data/
│   ├── __init__.py
│   └── tbd_dataset.py               # TBD/MSVD/MSR-VTT Dataset + DataLoader
│
├── utils/
│   ├── __init__.py
│   ├── vocab.py                     # 词表 & SCD 概念表构建
│   ├── feature_extractor.py         # ResNet-101 / C3D / Yolov5 / RoIAlign
│   └── metrics.py                   # BLEU/METEOR/ROUGE-L/CIDEr
│
└── scripts/
    ├── precompute_features.py       # 一次性提取并缓存视频特征
    ├── train.py                     # 训练入口
    └── evaluate.py                  # 评估/推理入口
```

## ⚙️ 关键超参数（论文 §4.2）

| 参数 | 值 | 说明 |
|---|---|---|
| `num_frames` | 26 | 均匀采样 |
| `max_boxes` | 5 | 每帧最多 5 个 bbox |
| `conf_threshold` | 0.5 | YOLOv5 置信度阈值 |
| `visual_out_dim` | 1024 | "feature size of all graph operations" |
| `word_embed_dim` | 512 | 词嵌入维度 |
| `num_concepts` | 300 (TBD/MSVD) / 400 (MSR-VTT) | SCD 候选概念数 |
| `d_model` | 512 | Decoder 维度 |
| `num_layers` | 6 | Decoder 层数 |
| `num_heads` | 10 | 注意力头数 |
| `max_caption_len` | 26 | 句子最大长度 |
| `min_word_freq` | 2 | 词表入选阈值 |
| `batch_size` | 64 | |
| `learning_rate` | 1e-6 | Adam 固定学习率 |
| `beam_size` | 6 | 推理 beam search |
| `β (loss)` | 0.4 | 公式 16 的长度权重 |
| `μ (位置阈值)` | 0.5 | TSI 中归一化坐标系下 |

## 🚀 使用流程

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备标注文件

`annotation_file` JSON 格式（任选其一）：

**格式 A**:
```json
{
  "videos": [
    {"video_id": "vid_0001", "caption": "the teacher stands at the front ...", "split": "train"},
    {"video_id": "vid_0001", "caption": "a teacher writes on the whiteboard", "split": "train"},
    {"video_id": "vid_0002", "caption": "...", "split": "val"}
  ]
}
```

**格式 B**:
```json
{
  "vid_0001": ["caption 1", "caption 2", ...],
  "vid_0002": [...]
}
```
配合 `xxx_train.txt` / `xxx_val.txt` / `xxx_test.txt` 提供 split。

### 3. 预提取特征（推荐，跑一次）
```bash
python scripts/precompute_features.py \
    --dataset TBD \
    --video_dir   data/videos/tbd \
    --output_dir  data/features/tbd \
    --annotation_file data/tbd_annotations.json \
    --c3d_weights /path/to/c3d_sports1m.pth
```

### 4. 训练
```bash
python scripts/train.py \
    --dataset TBD \
    --annotation_file data/tbd_annotations.json \
    --feature_dir data/features/tbd \
    --save_dir checkpoints/tbd_run1 \
    --num_epochs 100 \
    --batch_size 64 \
    --lr 1e-6
```
TensorBoard 日志写入 `logs/`，最佳模型按 `BLEU_4` 选择保存为 `best.pt`。

### 5. 评估 / 推理
```bash
python scripts/evaluate.py \
    --dataset TBD \
    --ckpt checkpoints/tbd_run1/best.pt \
    --feature_dir data/features/tbd \
    --annotation_file data/tbd_annotations.json \
    --split test \
    --output predictions_tbd.json \
    --beam_size 6
```

## 🧩 模块自测

每个模块文件都包含 `__main__` smoke-test，可单独运行验证形状：
```bash
python -m models.tsi_module
python -m models.ttc_module
python -m models.scd_module
python -m models.decoder
python -m models.stgin
python -m utils.feature_extractor
python -m utils.vocab
python -m utils.metrics
```

## 📊 论文结果（TBD 测试集，§4.3.1）

| Method | BLEU_4 | METEOR | ROUGE_L |
|---|---|---|---|
| HBI (Jin et al. 2024) | 33.45 | 18.22 | 41.41 |
| **STGIN (本复现目标)** | **31.68** | **18.44** | **41.85** |

## 📝 复现要点说明

1. **TSI 模块**：严格按公式 (1)–(7) 实现，含外观关系（指数化余弦）、位置关系（带阈值的指示函数）、归一化邻接矩阵的对称 GCN 更新。
2. **TTC 模块**：使用 2D+3D 拼接作为引导节点，通过加性注意力（公式 8）计算与每个教师候选 r_t^j 的相关性，softmax 加权（公式 9-10）。
3. **SCD 模块**：MLP 多标签分类，概念 embedding 由概率加权后送入 decoder 作为 cross-attention 的 K、V（论文 Eq. 12）。
4. **Decoder**：6 层、10 头，每层依次执行：自注意力 → 与 SCD 概念交叉注意力（公式 12）→ 与 v̂_s 和 v̂_a 双路视觉交叉注意力相加（公式 13）→ Add & Norm（公式 14）→ FFN。
5. **训练损失**：公式 (16) 的长度加权 CE + 公式 (11) 的 SCD BCE，β=0.4。
6. **推理**：beam_size=6，按 length-normalized log-prob 排序。

## 🔗 引用

```bibtex
@article{xiong2025stgin,
  title  = {Spatio-temporal graph interaction networks for teacher behavior description in classroom scene},
  author = {Xiong, Yu and He, Chengyang and Chen, Lulu and Cai, Ting},
  journal= {Engineering Applications of Artificial Intelligence},
  volume = {159},
  pages  = {111668},
  year   = {2025},
  doi    = {10.1016/j.engappai.2025.111668}
}
```
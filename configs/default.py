"""
STGIN 默认超参数 (论文 Section 4.2)
"""

class Config:
    # ----- 数据集 -----
    dataset_name    = 'TBD'
    annotation_file = 'data/annotations/tbd_annotations.json'
    feature_dir     = 'data/features/tbd'
    video_dir       = 'data/videos/tbd'       # 内含 clips_class_01/ ... clips_class_05/

    # 数据划分
    num_train_videos = 1200
    num_val_videos   = 400
    num_test_videos  = 400

    # ----- 视频采样 -----
    num_frames   = 26      # 均匀采样帧数
    frame_size   = 224
    clip_len_3d  = 16      # 3D CNN 输入 clip 长度
    clip_size_3d = 112

    # ----- 目标检测 -----
    max_boxes       = 5    # 每帧最多 N=5 个 bbox
    conf_threshold  = 0.5  # YOLOv5 置信度阈值

    # ----- 特征维度 -----
    obj_feat_dim    = 2048   # ResNet-101 layer4 通道数
    appear_dim      = 2048   # 2D CNN 输出
    motion_dim      = 4096   # 3D CNN fc6 输出
    visual_out_dim  = 1024   # 图操作统一维度
    concept_embed_dim = 512

    # ----- 词表/文本 -----
    word_embed_dim  = 512
    max_caption_len = 26
    min_word_freq   = 2
    vocab_size      = 4000

    # ----- SCD -----
    num_concepts = 300     # TBD=300, MSVD=300, MSR-VTT=400

    # ----- Transformer Decoder -----
    d_model    = 512
    num_layers = 6
    num_heads  = 8    # 论文原文10，但512/10不整除，改为8（512/8=64 ✓）
    d_ff       = 2048
    dropout    = 0.1

    # ----- TSI (GCN) -----
    num_gcn_layers  = 2
    dist_threshold  = 0.5

    # ----- 损失函数 -----
    beta_loss  = 0.4
    lambda_scd = 1.0

    # ----- 训练 -----
    batch_size    = 64
    learning_rate = 5e-4    # 论文用1e-6是基于20000条样本，小数据集需要调高
    optimizer     = 'adam'
    weight_decay  = 1e-5
    num_epochs    = 70
    warmup_steps  = 500     # 前N步线性升温学习率，防止初始崩溃
    label_smoothing = 0.1   # 平滑标签，缓解过拟合到单一token
    grad_clip     = 5.0
    log_interval  = 20
    save_interval = 5

    # ----- 推理 -----
    beam_size   = 6
    decode_mode = 'beam'

    # ----- I/O -----
    save_dir    = 'checkpoints'
    log_dir     = 'logs'
    device      = 'cuda'
    num_workers = 4
    seed        = 42

    # ----- 评估 -----
    best_metric = 'BLEU_4'


DATASET_CONFIGS = {
    'TBD':     {
        'num_concepts': 300, 'max_caption_len': 35,
        'annotation_file': 'data/annotations/tbd_annotations.json',
        'feature_dir': 'data/features/tbd',
        'video_dir':   'data/videos/tbd',
    },
    'MSVD':    {
        'num_concepts': 300, 'max_caption_len': 20,
        'annotation_file': 'data/annotations/msvd_annotations.json',
        'feature_dir': 'data/features/msvd',
        'video_dir':   'data/videos/msvd',
    },
    'MSR-VTT': {
        'num_concepts': 400, 'max_caption_len': 20,
        'annotation_file': 'data/annotations/msrvtt_annotations.json',
        'feature_dir': 'data/features/msrvtt',
        'video_dir':   'data/videos/msrvtt',
    },
}


def get_config(dataset_name: str = 'TBD') -> Config:
    cfg = Config()
    cfg.dataset_name = dataset_name
    if dataset_name in DATASET_CONFIGS:
        for k, v in DATASET_CONFIGS[dataset_name].items():
            setattr(cfg, k, v)
    return cfg
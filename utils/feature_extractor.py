"""
Feature Extraction
对应论文 Section 4.2 中的实现细节:
  - 每个视频均匀采样 26 帧 (key-frames)
  - 2D CNN: ResNet-101 (ImageNet 预训练)
  - 3D CNN: C3D (Sports1M 预训练), 输入 16 帧 clip
  - YOLOv5 检测目标, 每帧最多保留 top-5 (置信度≥0.5)
  - 教师候选: 通过 YOLOv5 检测后, 选 top-N 置信度最高的边界框
  - RoIAlign 从特征图中抠取对象级特征
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.ops import roi_align
import numpy as np


# -----------------------------------------------------------
# 1. 视频帧采样工具
# -----------------------------------------------------------
def uniform_sample_frames(video_path: str,
                          num_frames: int = 26,
                          resize: tuple = (224, 224)) -> torch.Tensor:
    """
    对视频均匀采样 num_frames 帧并 resize。
    使用 decord 优先，回退到 opencv。
    Returns:
      frames: (T, 3, H, W) float tensor, 已归一化到 [0,1]
    """
    try:
        import decord
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(video_path)
        total = len(vr)
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        frames = vr.get_batch(idxs)             # (T,H,W,3) uint8
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)     # (T,3,H,W)
    except Exception:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, total - 1, num_frames).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, f = cap.read()
            if not ret:
                f = np.zeros((resize[0], resize[1], 3), dtype=np.uint8)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()
        frames = np.stack(frames)               # (T,H,W,3)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

    # resize
    frames = F.interpolate(frames, size=resize, mode='bilinear', align_corners=False)
    return frames


# -----------------------------------------------------------
# 2. 2D CNN 帧级特征 (ResNet-101 倒数第二层 + avgpool)
# -----------------------------------------------------------
class ResNet101FeatureExtractor(nn.Module):
    """
    输出:
      - global_feat: (T, 2048)  avgpool 后的 v_t^a
      - feature_map: (T, 2048, 7, 7)  最后一个 conv block 的特征图 (供 RoIAlign 使用)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet101(weights=weights)
        # 去掉 fc
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # 用于 RoIAlign 的尺度: 输入 224 → 特征图 7×7, 故 spatial_scale = 1/32
        self.spatial_scale = 1.0 / 32.0

        # 标准 ImageNet 归一化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, frames: torch.Tensor):
        """
        frames: (T, 3, 224, 224)  ∈ [0,1]
        """
        # 归一化
        frames = torch.stack([self.normalize(f) for f in frames])

        x = self.stem(frames)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)              # (T, 2048, 7, 7)
        global_feat = self.avgpool(feat_map).flatten(1)  # (T, 2048)

        return global_feat, feat_map


# -----------------------------------------------------------
# 3. 3D CNN 帧级运动特征 (C3D)
# -----------------------------------------------------------
class C3D(nn.Module):
    """
    C3D (Tran et al. 2015), 4096 维 fc6 特征。
    输入: (B, 3, 16, 112, 112)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, 2)

        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, 2)

        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, 2)

        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool3d(2, 2, padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x));  x = self.pool1(x)
        x = self.relu(self.conv2(x));  x = self.pool2(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool3(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x)); x = self.pool4(x)
        x = self.relu(self.conv5a(x)); x = self.relu(self.conv5b(x)); x = self.pool5(x)
        x = x.flatten(1)
        x = self.relu(self.fc6(x))             # (B, 4096)
        return x


class C3DFeatureExtractor(nn.Module):
    """
    论文中: "For 3D CNN, we take 16-frame clips as input."
    输出每个 key-frame 的 4096 维运动特征。

    实现策略: 对 26 个采样 key-frame, 每个 key-frame 周围抽 16 帧形成 clip。
    """
    def __init__(self, weights_path: str = None):
        super().__init__()
        self.c3d = C3D()
        if weights_path and os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location='cpu')
            self.c3d.load_state_dict(sd, strict=False)

        self.mean = torch.tensor([90.0, 98.0, 102.0]).view(1, 3, 1, 1, 1)

    def forward(self, clips: torch.Tensor):
        """
        clips: (T, 3, 16, 112, 112)  T 个 16 帧 clip, 像素 ∈ [0,255]
        Returns:
          motion_feats: (T, 4096)
        """
        clips = clips - self.mean.to(clips.device)
        return self.c3d(clips)


def build_clips_from_frames(frames: torch.Tensor,
                            clip_len: int = 16,
                            clip_size: int = 112) -> torch.Tensor:
    """
    对每个 key-frame, 取其前后共 clip_len 帧作为 3D CNN 输入。
    frames: (T, 3, H, W)
    Returns:
      clips: (T, 3, clip_len, clip_size, clip_size)
    """
    T_ = frames.size(0)
    # 先把 frames resize 到 (clip_size, clip_size)
    frames_r = F.interpolate(frames, size=(clip_size, clip_size),
                             mode='bilinear', align_corners=False)
    frames_r = frames_r * 255.0  # C3D 通常输入 0-255

    half = clip_len // 2
    clips = []
    for t in range(T_):
        start = max(0, t - half)
        end = min(T_, start + clip_len)
        start = max(0, end - clip_len)
        clip = frames_r[start:end]              # (clip_len, 3, H, W)
        if clip.size(0) < clip_len:             # 用尾帧 pad
            pad = clip[-1:].repeat(clip_len - clip.size(0), 1, 1, 1)
            clip = torch.cat([clip, pad], dim=0)
        clip = clip.permute(1, 0, 2, 3)         # (3, clip_len, H, W)
        clips.append(clip)
    return torch.stack(clips, dim=0)


# -----------------------------------------------------------
# 4. YOLOv5 检测 + RoIAlign 抠取对象特征
# -----------------------------------------------------------
class YoloV5Detector:
    """
    论文: "Yolov5 is used to detect all the objects... where the number
           of bounding boxes N is limited to 5 in each frame.
           ... objects with confidence scores below 0.5 are filtered out."

    加载策略（按优先级）:
      1. ultralytics 新版包（pip install ultralytics）—— 无命名冲突，推荐
      2. torch.hub + 本地权重 —— 离线备用
      3. torch.hub 在线下载 —— 需要网络
    """
    def __init__(self,
                 model_name: str = 'yolov5s',
                 max_boxes: int = 5,
                 conf_thresh: float = 0.5,
                 person_class_id: int = 0,
                 device: str = 'cuda',
                 local_weights: str = None):
        self.max_boxes = max_boxes
        self.conf_thresh = conf_thresh
        self.person_class_id = person_class_id
        self.device = device
        self.available = False
        self._use_ultralytics = False  # 标记使用哪种 API

        weights_path = local_weights  # 本地权重（.pt 文件路径）

        # ── 方式1: ultralytics 新版包（推荐，无命名冲突）──
        ultralytics_err = None
        try:
            from ultralytics import YOLO
        except ImportError as e:
            ultralytics_err = f"ultralytics 包未安装 ({e})"
        else:
            try:
                if weights_path and os.path.exists(weights_path):
                    self.model = YOLO(weights_path)
                else:
                    self.model = YOLO(f'{model_name}.pt')
                self._use_ultralytics = True
                self.available = True
                print(f"[YoloV5Detector] ultralytics 加载成功: {weights_path or model_name}")
                return
            except Exception as e:
                ultralytics_err = f"ultralytics 加载权重失败: {type(e).__name__}: {e}"

        # ── 方式2/3: torch.hub（需隔离 sys.path 避免与项目 utils 冲突）──
        try:
            import sys, types
            # 临时屏蔽项目的 utils 包，避免与 yolov5 内部 utils 冲突
            _backup = sys.modules.pop('utils', None)
            _backup_subkeys = [k for k in sys.modules if k.startswith('utils.')]
            _backup_subs = {k: sys.modules.pop(k) for k in _backup_subkeys}

            try:
                import torch.hub
                if weights_path and os.path.exists(weights_path):
                    # 同时需要 yolov5 代码库
                    hub_dir = os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master')
                    if os.path.exists(hub_dir):
                        self.model = torch.hub.load(hub_dir, 'custom',
                                                    path=weights_path,
                                                    source='local', verbose=False)
                    else:
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                                    path=weights_path, verbose=False)
                    print(f"[YoloV5Detector] torch.hub 本地权重加载成功: {weights_path}")
                else:
                    self.model = torch.hub.load('ultralytics/yolov5', model_name,
                                                pretrained=True, verbose=False)
                    print(f"[YoloV5Detector] torch.hub 在线加载成功: {model_name}")
                self.model.conf = conf_thresh
                self.model.classes = [person_class_id]
                self.model.to(device).eval()
                self.available = True
                self._use_ultralytics = False
            finally:
                # 恢复项目 utils
                if _backup is not None:
                    sys.modules['utils'] = _backup
                sys.modules.update(_backup_subs)

        except Exception as e2:
            print(f"[YoloV5Detector] 所有加载方式均失败:")
            print(f"  ultralytics: {ultralytics_err}")
            print(f"  torch.hub  : {type(e2).__name__}: {e2}")
            print(f"  → obj_feats/teacher_feats 将为全零")
            self.available = False

    @torch.no_grad()
    def detect(self, frames: torch.Tensor):
        """
        frames: (T, 3, H, W) ∈ [0,1]
        Returns:
          boxes_per_frame: list of (M_t, 4)   bbox [x1,y1,x2,y2] 像素坐标
          confs_per_frame: list of (M_t,)
        """
        if not self.available:
            T_ = frames.size(0)
            return [torch.zeros(0, 4)] * T_, [torch.zeros(0)] * T_

        H, W = frames.shape[-2:]
        frames_np = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')

        boxes_list, confs_list = [], []

        if self._use_ultralytics:
            # ultralytics YOLO API
            results = self.model.predict(
                [f for f in frames_np],
                conf=self.conf_thresh,
                classes=[self.person_class_id],
                verbose=False,
                device=self.device,
            )
            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    boxes_list.append(torch.zeros(0, 4))
                    confs_list.append(torch.zeros(0))
                    continue
                xyxy  = boxes.xyxy.cpu()    # (M, 4)
                confs = boxes.conf.cpu()    # (M,)
                # 按置信度排序，保留 top max_boxes
                idx = confs.argsort(descending=True)[:self.max_boxes]
                boxes_list.append(xyxy[idx])
                confs_list.append(confs[idx])
        else:
            # torch.hub API (老版 yolov5)
            results = self.model([f for f in frames_np], size=max(H, W))
            for det in results.xyxy:
                det = det.cpu()
                if det.numel() == 0:
                    boxes_list.append(torch.zeros(0, 4))
                    confs_list.append(torch.zeros(0))
                    continue
                det = det[det[:, 4] >= self.conf_thresh]
                det = det[det[:, 4].argsort(descending=True)][:self.max_boxes]
                boxes_list.append(det[:, :4])
                confs_list.append(det[:, 4])

        return boxes_list, confs_list


def extract_roi_features(feature_map: torch.Tensor,
                         boxes_per_frame: list,
                         spatial_scale: float,
                         output_size: int = 7,
                         max_boxes: int = 5) -> tuple:
    """
    用 RoIAlign 从 ResNet-101 的特征图中抠出每个 bbox 的对象特征,
    再 avgpool 成向量。

    Args:
      feature_map:     (T, C, h, w)   ResNet feature map
      boxes_per_frame: list of (M_t, 4)  在原图 (224x224) 上的坐标
      spatial_scale:   原图坐标 → feature_map 坐标的比例
      output_size:     RoIAlign 输出空间尺寸
    Returns:
      obj_feats: (T, max_boxes, C)
      obj_boxes: (T, max_boxes, 4)  归一化到 [0,1]
      obj_mask:  (T, max_boxes)
    """
    T_, C, h, w = feature_map.shape
    device = feature_map.device

    # 把 boxes 整理成 RoIAlign 所需格式: (K, 5) [batch_idx, x1, y1, x2, y2]
    rois = []
    box_tensor = torch.zeros(T_, max_boxes, 4, device=device)
    mask_tensor = torch.zeros(T_, max_boxes, device=device)

    for t, boxes in enumerate(boxes_per_frame):
        boxes = boxes.to(device)
        M = min(boxes.size(0), max_boxes)
        if M > 0:
            box_tensor[t, :M] = boxes[:M] / 224.0     # 归一化 (原图尺寸 224)
            mask_tensor[t, :M] = 1.0
            batch_idx = torch.full((M, 1), t, device=device, dtype=boxes.dtype)
            rois.append(torch.cat([batch_idx, boxes[:M]], dim=1))

    if not rois:
        obj_feats = torch.zeros(T_, max_boxes, C, device=device)
        return obj_feats, box_tensor, mask_tensor

    rois = torch.cat(rois, dim=0)                     # (K, 5)
    pooled = roi_align(feature_map, rois,
                       output_size=(output_size, output_size),
                       spatial_scale=spatial_scale,
                       aligned=True)                  # (K, C, 7, 7)
    pooled = pooled.mean(dim=(-1, -2))                # (K, C)

    obj_feats = torch.zeros(T_, max_boxes, C, device=device)
    cursor = 0
    for t, boxes in enumerate(boxes_per_frame):
        M = min(boxes.size(0), max_boxes)
        if M > 0:
            obj_feats[t, :M] = pooled[cursor:cursor + M]
            cursor += M

    return obj_feats, box_tensor, mask_tensor


# -----------------------------------------------------------
# 5. 一站式视频特征提取 pipeline
# -----------------------------------------------------------
class VideoFeaturePipeline(nn.Module):
    """
    对单个视频, 一次性提取 STGIN 所需的全部输入特征:
      - obj_feats / obj_boxes / obj_mask
      - appear_feats (ResNet)
      - motion_feats (C3D)
      - teacher_feats / teacher_mask (top-N 教师候选)
    """
    def __init__(self,
                 num_frames: int = 26,
                 max_boxes: int = 5,
                 device: str = 'cuda',
                 c3d_weights: str = None,
                 yolo_weights: str = None,   # 本地 YOLOv5 权重路径
                 use_yolo: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.max_boxes = max_boxes
        self.device = device

        self.resnet = ResNet101FeatureExtractor(pretrained=True).to(device).eval()
        self.c3d = C3DFeatureExtractor(weights_path=c3d_weights).to(device).eval()

        self.detector = YoloV5Detector(
            max_boxes=max_boxes, conf_thresh=0.5, device=device,
            local_weights=yolo_weights,
        ) if use_yolo else None

    @torch.no_grad()
    def forward(self, video_path: str):
        # 1) 帧采样
        frames = uniform_sample_frames(video_path, self.num_frames, (224, 224))
        frames = frames.to(self.device)

        # 2) 2D 特征
        appear_feats, feat_map = self.resnet(frames)          # (T,2048), (T,2048,7,7)

        # 3) 3D 特征
        clips = build_clips_from_frames(frames, clip_len=16, clip_size=112)
        motion_feats = self.c3d(clips.to(self.device))         # (T, 4096)

        # 4) 目标检测 + RoIAlign
        if self.detector is not None:
            boxes_pf, confs_pf = self.detector.detect(frames)
        else:
            boxes_pf = [torch.zeros(0, 4)] * self.num_frames
            confs_pf = [torch.zeros(0)] * self.num_frames

        obj_feats, obj_boxes, obj_mask = extract_roi_features(
            feat_map, boxes_pf,
            spatial_scale=self.resnet.spatial_scale,
            output_size=7, max_boxes=self.max_boxes,
        )

        # 5) 教师候选 = 同一组 (TTC 中也用 top-N 置信度的 bbox)
        teacher_feats = obj_feats.clone()
        teacher_mask = obj_mask.clone()

        return {
            'appear_feats':  appear_feats,         # (T, 2048)
            'motion_feats':  motion_feats,         # (T, 4096)
            'obj_feats':     obj_feats,            # (T, N, 2048)
            'obj_boxes':     obj_boxes,            # (T, N, 4)  ∈ [0,1]
            'obj_mask':      obj_mask,             # (T, N)
            'teacher_feats': teacher_feats,        # (T, N, 2048)
            'teacher_mask':  teacher_mask,         # (T, N)
        }


if __name__ == "__main__":
    # 自测 (无需真实视频)
    extractor = ResNet101FeatureExtractor(pretrained=False)
    x = torch.rand(4, 3, 224, 224)
    g, fm = extractor(x)
    print("ResNet global:", g.shape, " feat_map:", fm.shape)

    c3d = C3D()
    y = torch.randn(2, 3, 16, 112, 112)
    print("C3D fc6:", c3d(y).shape)
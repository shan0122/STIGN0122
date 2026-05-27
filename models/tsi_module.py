"""
Teacher-Student Spatial Interaction (TSI) Module
对应论文 Section 3.1

功能：
1. 使用 Yolov5 在每个 key-frame 中检测人体边界框
2. 通过 RoIAlign 提取每个 bbox 区域的特征 s_k^i
3. 计算 appearance relationship R_a (公式 1) 与 positional relationship R_s (公式 2)
4. 构建关系图 G_ij (公式 3-5)
5. 通过 GCN 迭代更新节点表征 (公式 6)
6. mean-pooling 得到帧级空间交互特征 v̂_s (公式 7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInteractionGraph(nn.Module):
    """
    构建师生空间交互图，并通过 GCN 进行迭代更新。
    输入:
      - obj_feats:  (B, N_frame, N_obj, D)  每帧中 N 个目标的 RoIAlign 特征
      - obj_boxes:  (B, N_frame, N_obj, 4)  每帧中 N 个目标的 bbox [x1,y1,x2,y2]
      - obj_mask:   (B, N_frame, N_obj)     有效目标 mask
    输出:
      - v_s_hat:    (B, N_frame, D)         帧级空间交互特征
    """

    def __init__(self,
                 feat_dim: int = 1024,
                 num_gcn_layers: int = 2,
                 dist_threshold: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_gcn_layers = num_gcn_layers
        self.dist_threshold = dist_threshold  # 论文中位置阈值 μ
        self.eps = eps

        # GCN 各层权重 W^(l) ∈ R^{d×d}
        self.gcn_weights = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim, bias=False)
            for _ in range(num_gcn_layers)
        ])

        # 节点特征归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(feat_dim) for _ in range(num_gcn_layers)
        ])

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.gcn_weights:
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _box_centers(boxes: torch.Tensor) -> torch.Tensor:
        """bbox [x1,y1,x2,y2] → 中心点 (cx, cy)"""
        cx = (boxes[..., 0] + boxes[..., 2]) * 0.5
        cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
        return torch.stack([cx, cy], dim=-1)

    def appearance_relation(self, feats: torch.Tensor) -> torch.Tensor:
        """
        公式 (1):
        R_a(x_i^a, x_j^a) = exp( cos(x_i, x_j) / max_{i,j} cos(x_i, x_j) )

        feats: (B, F, N, D)
        return: (B, F, N, N)
        """
        # 单位化
        feats_norm = F.normalize(feats, p=2, dim=-1)
        # 余弦相似度
        cos_sim = torch.matmul(feats_norm, feats_norm.transpose(-1, -2))  # (B,F,N,N)
        # 每帧内的最大值（论文中 max_{i,j}）
        max_val = cos_sim.amax(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        R_a = torch.exp(cos_sim / max_val)
        return R_a

    def positional_relation(self,
                            boxes: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """
        公式 (2):
        R_s(x_i^s, x_j^s) = I( d(x_i^s, x_j^s) < μ )

        boxes: (B, F, N, 4)
        mask:  (B, F, N)
        return: (B, F, N, N)
        """
        centers = self._box_centers(boxes)  # (B,F,N,2)
        diff = centers.unsqueeze(-2) - centers.unsqueeze(-3)  # (B,F,N,N,2)
        dist = torch.norm(diff, p=2, dim=-1)  # (B,F,N,N)

        # 用图像对角线归一化距离 (假设 bbox 已归一化到 [0,1])
        R_s = (dist < self.dist_threshold).float()

        # mask 掉无效节点
        m = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B,F,N,N)
        R_s = R_s * m
        return R_s

    def build_graph(self,
                    feats: torch.Tensor,
                    boxes: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """
        公式 (3)-(5):
        G_ij = (1/Z_i) * R_s(x_i^s, x_j^s) * g( R_a(x_i^a, x_j^a) )
        其中 g(·) = ReLU, Z_i 归一化使每行和为 1
        """
        R_a = self.appearance_relation(feats)        # (B,F,N,N)
        R_s = self.positional_relation(boxes, mask)  # (B,F,N,N)

        # g(R_a) = ReLU(R_a)
        g_Ra = F.relu(R_a)

        # 未归一化的边权
        edges = R_s * g_Ra  # (B,F,N,N)

        # 归一化因子 Z_i (公式 5)
        Z_i = edges.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        G = edges / Z_i  # (B,F,N,N)

        return G

    @staticmethod
    def _normalize_adj(G: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        对应公式 (6) 中的 Λ^{-1/2} G Λ^{-1/2}，对称归一化邻接矩阵
        Λ_ii = Σ_j G_ij
        """
        deg = G.sum(dim=-1).clamp(min=eps)        # (B,F,N)
        deg_inv_sqrt = deg.pow(-0.5)
        # 构造对角矩阵并左右乘
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)  # (B,F,N,N)
        return D_inv_sqrt @ G @ D_inv_sqrt

    def gcn_forward(self,
                    H: torch.Tensor,
                    G: torch.Tensor) -> torch.Tensor:
        """
        公式 (6):
        H^(l+1) = ReLU( H^(l) + Λ^{-1/2} G Λ^{-1/2} H^(l) W^(l) )
        """
        G_hat = self._normalize_adj(G)  # (B,F,N,N)

        for l in range(self.num_gcn_layers):
            # 邻居信息聚合
            aggregated = torch.matmul(G_hat, H)         # (B,F,N,D)
            transformed = self.gcn_weights[l](aggregated)  # (B,F,N,D)
            # 残差 + ReLU + LayerNorm
            H = F.relu(H + transformed)
            H = self.layer_norms[l](H)
        return H

    def forward(self,
                obj_feats: torch.Tensor,
                obj_boxes: torch.Tensor,
                obj_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          obj_feats: (B, F, N, D)
          obj_boxes: (B, F, N, 4)  bbox 已归一化到 [0,1]
          obj_mask:  (B, F, N)
        Returns:
          v_s_hat:   (B, F, D)
        """
        # 1) 构建图
        G = self.build_graph(obj_feats, obj_boxes, obj_mask)

        # 2) GCN 迭代
        H = self.gcn_forward(obj_feats, G)  # (B,F,N,D)

        # 3) 公式 (7): mean-pooling 得到帧级特征
        mask_f = obj_mask.unsqueeze(-1).float()       # (B,F,N,1)
        H_masked = H * mask_f
        num_valid = mask_f.sum(dim=-2).clamp(min=1.0)  # (B,F,1)
        v_s_hat = H_masked.sum(dim=-2) / num_valid     # (B,F,D)

        return v_s_hat


class TSIModule(nn.Module):
    """
    TSI 整体封装，可在外部独立使用。
    """
    def __init__(self,
                 feat_dim: int = 1024,
                 num_gcn_layers: int = 2,
                 dist_threshold: float = 0.5):
        super().__init__()
        self.graph = SpatialInteractionGraph(
            feat_dim=feat_dim,
            num_gcn_layers=num_gcn_layers,
            dist_threshold=dist_threshold,
        )

    def forward(self, obj_feats, obj_boxes, obj_mask):
        return self.graph(obj_feats, obj_boxes, obj_mask)


if __name__ == "__main__":
    # 简单自测
    B, F_, N, D = 2, 26, 5, 1024
    feats = torch.randn(B, F_, N, D)
    boxes = torch.rand(B, F_, N, 4)
    # 保证 x2>x1, y2>y1
    boxes[..., 2] = boxes[..., 0] + 0.1
    boxes[..., 3] = boxes[..., 1] + 0.1
    mask = torch.ones(B, F_, N)

    tsi = TSIModule(feat_dim=D)
    out = tsi(feats, boxes, mask)
    print("TSI output:", out.shape)   # 期望 (2, 26, 1024)
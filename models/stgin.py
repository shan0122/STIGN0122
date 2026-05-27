"""
STGIN: Spatio-Temporal Graph Interaction Network
对应论文 Fig. 2 整体架构

主流程:
  Video Frames + bbox →
    ┌── TSI  → v̂_s (B, F, D)
    ├── TTC  → v̂_a (B, F, D)
    └── SCD  → q, C  (语义概念)
              ↓
       Description Generator → caption
"""

import torch
import torch.nn as nn

from models.tsi_module import TSIModule
from models.ttc_module import TTCModule
from models.scd_module import SemanticConceptDetector
from models.decoder import DescriptionGenerator, LengthWeightedCELoss


class STGIN(nn.Module):
    """
    Args:
      vocab_size:          词表大小
      num_concepts:        SCD 候选语义概念数量 K
      obj_feat_dim:        RoIAlign 后的对象特征维度
      appear_dim:          ResNet-101 提取的 2D 外观特征维度
      motion_dim:          C3D 提取的 3D 运动特征维度
      visual_out_dim:      TSI/TTC 输出的统一可视特征维度
      concept_embed_dim:   SCD 概念 embedding 维度
      d_model:             Transformer decoder 维度
      num_layers:          decoder 层数 (论文: 6)
      num_heads:           多头注意力头数 (论文: 10)
      max_len:             生成序列最大长度 (论文: 26)
      pad_id/bos_id/eos_id:特殊 token id
      num_gcn_layers:      TSI 中 GCN 层数
      dist_threshold:      TSI 中位置阈值 μ
      beta:                损失函数中长度权重超参数 (论文: 0.4)
    """

    def __init__(self,
                 vocab_size: int,
                 num_concepts: int = 300,
                 obj_feat_dim: int = 1024,
                 appear_dim: int = 2048,
                 motion_dim: int = 4096,
                 visual_out_dim: int = 1024,
                 concept_embed_dim: int = 512,
                 d_model: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 10,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 26,
                 pad_id: int = 0,
                 bos_id: int = 1,
                 eos_id: int = 2,
                 num_gcn_layers: int = 2,
                 dist_threshold: float = 0.5,
                 beta: float = 0.4,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_len = max_len

        # === Module 1: TSI ===
        self.tsi = TSIModule(
            feat_dim=obj_feat_dim,
            num_gcn_layers=num_gcn_layers,
            dist_threshold=dist_threshold,
        )
        # TSI 输出 obj_feat_dim → visual_out_dim 的投影 (若不一致)
        self.proj_tsi_out = (
            nn.Linear(obj_feat_dim, visual_out_dim)
            if obj_feat_dim != visual_out_dim else nn.Identity()
        )

        # === Module 2: TTC ===
        self.ttc = TTCModule(
            teacher_feat_dim=obj_feat_dim,
            appear_dim=appear_dim,
            motion_dim=motion_dim,
            hidden_dim=512,
            out_dim=visual_out_dim,
        )

        # === Module 3: SCD ===
        self.scd = SemanticConceptDetector(
            input_dim=appear_dim + motion_dim,
            num_concepts=num_concepts,
            embed_dim=concept_embed_dim,
            hidden_dim=1024,
            dropout=0.5,
        )

        # === Module 4: Description Generator ===
        self.decoder = DescriptionGenerator(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_id=pad_id,
            visual_dim=visual_out_dim,
            concept_dim=concept_embed_dim,
        )

        # === 损失函数 ===
        self.caption_loss = LengthWeightedCELoss(
            pad_id=pad_id, beta=beta,
            label_smoothing=label_smoothing,
        )

    # -----------------------------------------------------------
    # 编码器: 把视觉特征转成 v̂_s, v̂_a, C
    # -----------------------------------------------------------
    def encode(self,
               obj_feats: torch.Tensor,
               obj_boxes: torch.Tensor,
               obj_mask: torch.Tensor,
               appear_feats: torch.Tensor,
               motion_feats: torch.Tensor,
               teacher_feats: torch.Tensor,
               teacher_mask: torch.Tensor):
        """
        Args:
          obj_feats:    (B, F, N, D_r)   TSI 中的所有目标 RoIAlign 特征 s_k^i
          obj_boxes:    (B, F, N, 4)     TSI 中的 bbox (归一化到 [0,1])
          obj_mask:     (B, F, N)
          appear_feats: (B, F, D_a)
          motion_feats: (B, F, D_m)
          teacher_feats:(B, F, N_t, D_r) TTC 中筛出的 top-N 教师候选特征 r_t^j
          teacher_mask: (B, F, N_t)
        Returns:
          v_s_hat: (B, F, visual_out_dim)
          v_a_hat: (B, F, visual_out_dim)
          q:       (B, K)
          C:       (B, K, concept_embed_dim)
        """
        # --- TSI ---
        v_s_hat = self.tsi(obj_feats, obj_boxes, obj_mask)   # (B,F,D_r)
        v_s_hat = self.proj_tsi_out(v_s_hat)                  # (B,F,visual_out_dim)

        # --- TTC ---
        v_a_hat = self.ttc(appear_feats, motion_feats,
                           teacher_feats, teacher_mask)       # (B,F,visual_out_dim)

        # --- SCD ---
        # 全局视频特征 = 各帧 [v_t^a || v_t^m] 在时间维上做平均 (论文 Fig. 2 中的 Average)
        global_feat = torch.cat([appear_feats, motion_feats], dim=-1).mean(dim=1)  # (B, D_a+D_m)
        q, C = self.scd(global_feat)                          # (B,K), (B,K,E)

        return v_s_hat, v_a_hat, q, C

    # -----------------------------------------------------------
    # 训练 forward (teacher forcing)
    # -----------------------------------------------------------
    def forward(self,
                batch: dict,
                return_loss: bool = True):
        """
        训练时输入一个 batch dict，含以下键：
          - obj_feats, obj_boxes, obj_mask
          - appear_feats, motion_feats
          - teacher_feats, teacher_mask
          - caption_in:  (B, T)  解码器输入 (前移一位，含 <bos>)
          - caption_tgt: (B, T)  解码器目标 (后移一位，含 <eos>)
          - concept_labels: (B, K)  SCD 多标签 ground-truth
          - lambda_scd: float  SCD loss 权重 (可选)
        """
        v_s, v_a, q, C = self.encode(
            obj_feats=batch['obj_feats'],
            obj_boxes=batch['obj_boxes'],
            obj_mask=batch['obj_mask'],
            appear_feats=batch['appear_feats'],
            motion_feats=batch['motion_feats'],
            teacher_feats=batch['teacher_feats'],
            teacher_mask=batch['teacher_mask'],
        )

        logits = self.decoder(batch['caption_in'], v_s, v_a, C)  # (B,T,V)

        if not return_loss:
            return logits, q

        # caption loss (公式 16)
        loss_cap = self.caption_loss(logits, batch['caption_tgt'])

        # SCD loss (公式 11)
        loss_scd = self.scd.bce_loss(q, batch['concept_labels'])

        lambda_scd = batch.get('lambda_scd', 1.0)
        loss = loss_cap + lambda_scd * loss_scd

        return {
            'loss': loss,
            'loss_cap': loss_cap.detach(),
            'loss_scd': loss_scd.detach(),
            'logits': logits,
            'concept_probs': q.detach(),
        }

    # -----------------------------------------------------------
    # 推理 (greedy / beam search)
    # -----------------------------------------------------------
    @torch.no_grad()
    def generate(self,
                 batch: dict,
                 mode: str = 'beam',
                 beam_size: int = 6):
        v_s, v_a, _, C = self.encode(
            obj_feats=batch['obj_feats'],
            obj_boxes=batch['obj_boxes'],
            obj_mask=batch['obj_mask'],
            appear_feats=batch['appear_feats'],
            motion_feats=batch['motion_feats'],
            teacher_feats=batch['teacher_feats'],
            teacher_mask=batch['teacher_mask'],
        )

        if mode == 'greedy':
            return self.decoder.greedy_decode(
                v_s, v_a, C, self.bos_id, self.eos_id, self.max_len)
        elif mode == 'beam':
            return self.decoder.beam_search(
                v_s, v_a, C, self.bos_id, self.eos_id,
                beam_size=beam_size, max_len=self.max_len)
        else:
            raise ValueError(f"Unknown decode mode: {mode}")


if __name__ == "__main__":
    # ---- smoke test ----
    B, F_, N, N_t = 2, 26, 5, 5
    V, K = 1000, 300
    D_r, D_a, D_m = 1024, 2048, 4096
    T = 20

    model = STGIN(
        vocab_size=V,
        num_concepts=K,
        obj_feat_dim=D_r,
        appear_dim=D_a,
        motion_dim=D_m,
        visual_out_dim=1024,
        concept_embed_dim=512,
        d_model=512,
        num_layers=2,   # 测试用小一点
        num_heads=8,
        max_len=T,
    )

    batch = {
        'obj_feats':     torch.randn(B, F_, N, D_r),
        'obj_boxes':     torch.rand(B, F_, N, 4),
        'obj_mask':      torch.ones(B, F_, N),
        'appear_feats':  torch.randn(B, F_, D_a),
        'motion_feats':  torch.randn(B, F_, D_m),
        'teacher_feats': torch.randn(B, F_, N_t, D_r),
        'teacher_mask':  torch.ones(B, F_, N_t),
        'caption_in':    torch.randint(1, V, (B, T)),
        'caption_tgt':   torch.randint(1, V, (B, T)),
        'concept_labels':(torch.rand(B, K) > 0.7).float(),
    }
    # 确保 boxes 有效
    batch['obj_boxes'][..., 2] = batch['obj_boxes'][..., 0] + 0.1
    batch['obj_boxes'][..., 3] = batch['obj_boxes'][..., 1] + 0.1

    out = model(batch)
    print("Train loss:", out['loss'].item(),
          " cap:", out['loss_cap'].item(),
          " scd:", out['loss_scd'].item())
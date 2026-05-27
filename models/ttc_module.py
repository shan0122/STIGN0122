"""
Teacher Temporal Context (TTC) Module
对应论文 Section 3.2

功能：
1. 通过目标检测器从每个 key-frame 选出 top-N 置信度的"教师"候选 (r_t^j)
2. 用 2D CNN / 3D CNN 提取帧级外观特征 v_t^a 与运动特征 v_t^m
3. 拼接为引导节点 v_t = [v_t^a || v_t^m]
4. 通过 (公式 8) 计算 v_t 与每个 r_t^j 的相关性 e_tj
5. softmax 归一化得 a_tj (公式 9)
6. 通过 (公式 10) 得到增强的对象特征 v̂_a
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherTemporalContext(nn.Module):
    """
    Args:
      teacher_feat_dim:  r_t^j 的维度 (与 TSI 中 obj_feats 维度一致, e.g. 1024)
      appear_dim:        v_t^a 维度 (2D CNN 输出, e.g. 2048)
      motion_dim:        v_t^m 维度 (3D CNN 输出, e.g. 4096)
      hidden_dim:        投影后的统一维度
      out_dim:           v̂_a 输出维度
    """

    def __init__(self,
                 teacher_feat_dim: int = 1024,
                 appear_dim: int = 2048,
                 motion_dim: int = 4096,
                 hidden_dim: int = 512,
                 out_dim: int = 1024):
        super().__init__()
        self.teacher_feat_dim = teacher_feat_dim
        self.appear_dim = appear_dim
        self.motion_dim = motion_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 帧级引导特征融合: v_t = [v_t^a || v_t^m] ∈ R^{D_f}
        self.frame_dim = appear_dim + motion_dim  # D_f

        # 公式 (8) 中的可学习参数 W_t, W_r, W_f
        self.W_t = nn.Linear(self.frame_dim, hidden_dim, bias=False)
        self.W_r = nn.Linear(teacher_feat_dim, hidden_dim, bias=False)
        self.W_f = nn.Linear(2 * hidden_dim, 1, bias=False)

        # 公式 (10) 中的 W (对 r_t^j 的变换) 与 W_attn
        self.W = nn.Linear(teacher_feat_dim, self.frame_dim, bias=False)
        self.W_attn = nn.Linear(self.frame_dim, out_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in [self.W_t, self.W_r, self.W_f, self.W, self.W_attn]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                appear_feats: torch.Tensor,
                motion_feats: torch.Tensor,
                teacher_feats: torch.Tensor,
                teacher_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          appear_feats:  (B, F, D_a)        2D CNN 提取的帧级外观特征 v_t^a
          motion_feats:  (B, F, D_m)        3D CNN 提取的帧级运动特征 v_t^m
          teacher_feats: (B, F, N, D_r)     每帧的 N 个教师候选特征 r_t^j
          teacher_mask:  (B, F, N) or None  有效 mask
        Returns:
          v_a_hat:       (B, F, out_dim)    时序上下文增强后的对象特征 v̂_a
        """
        B, F_, N, D_r = teacher_feats.shape
        device = teacher_feats.device

        if teacher_mask is None:
            teacher_mask = torch.ones(B, F_, N, device=device)

        # 1) 拼接帧级引导特征 v_t = [v_t^a || v_t^m]
        v_t = torch.cat([appear_feats, motion_feats], dim=-1)  # (B, F, D_f)

        # 2) 公式 (8): e_tj = σ( W_f [W_t v_t || W_r r_t^j] )
        Wt_vt = self.W_t(v_t)                       # (B, F, H)
        Wr_rt = self.W_r(teacher_feats)             # (B, F, N, H)

        # 把 Wt_vt 广播到与 Wr_rt 相同的形状
        Wt_vt_exp = Wt_vt.unsqueeze(2).expand(-1, -1, N, -1)  # (B, F, N, H)
        concat = torch.cat([Wt_vt_exp, Wr_rt], dim=-1)        # (B, F, N, 2H)

        e_tj = self.W_f(concat).squeeze(-1)                   # (B, F, N)
        e_tj = torch.tanh(e_tj)  # 论文中 σ(·) 为激活函数

        # 3) 公式 (9): a_tj = softmax_k(e_tk)，仅对有效目标做 softmax
        mask = teacher_mask.bool()
        neg_inf = torch.finfo(e_tj.dtype).min
        e_tj_masked = e_tj.masked_fill(~mask, neg_inf)
        a_tj = F.softmax(e_tj_masked, dim=-1)                 # (B, F, N)
        # 处理整帧无效的情况 → 置 0
        a_tj = torch.where(mask.any(dim=-1, keepdim=True),
                           a_tj, torch.zeros_like(a_tj))

        # 4) 公式 (10): v̂_a = W_attn ( v_t + Σ_j a_tj * r_t^j * W )
        rW = self.W(teacher_feats)                            # (B, F, N, D_f)
        weighted = (a_tj.unsqueeze(-1) * rW).sum(dim=-2)      # (B, F, D_f)
        fused = v_t + weighted                                # (B, F, D_f)
        v_a_hat = self.W_attn(fused)                          # (B, F, out_dim)

        return v_a_hat


class TTCModule(nn.Module):
    """TTC 整体封装"""

    def __init__(self,
                 teacher_feat_dim: int = 1024,
                 appear_dim: int = 2048,
                 motion_dim: int = 4096,
                 hidden_dim: int = 512,
                 out_dim: int = 1024):
        super().__init__()
        self.ttc = TeacherTemporalContext(
            teacher_feat_dim=teacher_feat_dim,
            appear_dim=appear_dim,
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, appear_feats, motion_feats, teacher_feats, teacher_mask=None):
        return self.ttc(appear_feats, motion_feats, teacher_feats, teacher_mask)


if __name__ == "__main__":
    B, F_, N = 2, 26, 5
    D_r, D_a, D_m = 1024, 2048, 4096

    appear = torch.randn(B, F_, D_a)
    motion = torch.randn(B, F_, D_m)
    teacher = torch.randn(B, F_, N, D_r)
    mask = torch.ones(B, F_, N)

    ttc = TTCModule(teacher_feat_dim=D_r, appear_dim=D_a, motion_dim=D_m,
                    hidden_dim=512, out_dim=1024)
    out = ttc(appear, motion, teacher, mask)
    print("TTC output:", out.shape)  # 期望 (2, 26, 1024)
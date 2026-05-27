"""
Description Generator (Transformer-based Decoder)
对应论文 Section 3.3

功能：
1. 输入: 已生成的词序列 w_i, 语义概念 C, 空间特征 v̂_s, 时序特征 v̂_a
2. 公式 (12): Z_i^l = LN( w_i + MHA(w_i, C, C) )     -- 词 vs 语义概念交叉注意力
3. 公式 (13): H_i^l = MHA(Z_i^l, v̂_s, v̂_s) + MHA(Z_i^l, v̂_a, v̂_a)
4. 公式 (14): Y_i^l = LN(H_i^l + Z_i^l)
5. 经过 FeedForward 后输出, 堆叠 N 层
6. 公式 (15): p_voc = softmax(W_voc Y_d + b_voc)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """标准 Multi-Head Attention"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除。" \
            f"请在 configs/default.py 中调整，推荐: 512/8, 512/4, 256/8"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B = Q.size(0)
        Tq = Q.size(1)
        Tk = K.size(1)

        q = self.W_q(Q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,d_k)
        k = self.W_k(K).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)

        if mask is not None:
            # mask: broadcastable to (B,H,Tq,Tk)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                 # (B,H,Tq,d_k)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.W_o(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class STGINDecoderLayer(nn.Module):
    """
    单层 STGIN decoder，严格按照公式 (12)-(14) 实现。
    """

    def __init__(self,
                 d_model: int = 512,
                 num_heads: int = 10,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        # 词 self-attention (带 causal mask)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 公式 (12): MHA(w_i, C, C)
        self.concept_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 公式 (13): MHA(Z_i, v̂_s, v̂_s) 与 MHA(Z_i, v̂_a, v̂_a)
        self.spatial_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.temporal_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ff = FeedForward(d_model, d_ff, dropout)

        self.ln1 = nn.LayerNorm(d_model)   # 自注意力后
        self.ln2 = nn.LayerNorm(d_model)   # 公式 (12) 之后 (Z_i)
        self.ln3 = nn.LayerNorm(d_model)   # 公式 (14) 之后 (Y_i)
        self.ln4 = nn.LayerNorm(d_model)   # FFN 之后

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                w: torch.Tensor,
                C: torch.Tensor,
                v_s: torch.Tensor,
                v_a: torch.Tensor,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          w:        (B, T, D)  当前层输入词序列表征
          C:        (B, K, D)  SCD 输出的语义概念 embedding
          v_s:      (B, F, D)  TSI 空间交互特征
          v_a:      (B, F, D)  TTC 时序上下文特征
          tgt_mask: (1,1,T,T)  causal mask
        Returns:
          out:      (B, T, D)
        """
        # 0) 词自注意力 (causal)
        sa = self.self_attn(w, w, w, mask=tgt_mask)
        w = self.ln1(w + self.dropout(sa))

        # 1) 公式 (12): Z_i = LN(w + MHA(w, C, C))
        ca = self.concept_attn(w, C, C)
        Z = self.ln2(w + self.dropout(ca))

        # 2) 公式 (13): H = MHA(Z, v̂_s, v̂_s) + MHA(Z, v̂_a, v̂_a)
        s_attn = self.spatial_attn(Z, v_s, v_s)
        t_attn = self.temporal_attn(Z, v_a, v_a)
        H = s_attn + t_attn

        # 3) 公式 (14): Y = LN(H + Z)
        Y = self.ln3(H + Z)

        # 4) FFN + Add & Norm
        out = self.ln4(Y + self.dropout(self.ff(Y)))
        return out


class DescriptionGenerator(nn.Module):
    """
    完整的描述生成器:
      - 词 embedding + 位置编码
      - N 层 STGINDecoderLayer
      - 公式 (15): p_voc = softmax(W_voc Y_d + b_voc)
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 10,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 26,
                 pad_id: int = 0,
                 # 三类外部特征的输入维度
                 visual_dim: int = 1024,
                 concept_dim: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_len = max_len

        # 词 embedding
        self.word_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 2)

        # 将不同来源的特征投影到 d_model
        self.proj_vs = nn.Linear(visual_dim, d_model)
        self.proj_va = nn.Linear(visual_dim, d_model)
        self.proj_C  = nn.Linear(concept_dim, d_model)

        self.layers = nn.ModuleList([
            STGINDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 公式 (15)
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_causal_mask(T: int, device) -> torch.Tensor:
        """生成下三角 causal mask，形状 (1,1,T,T)"""
        mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.uint8))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

    def forward(self,
                tgt_tokens: torch.Tensor,
                v_s: torch.Tensor,
                v_a: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        训练时使用 teacher forcing 一次性 forward 所有时间步。

        Args:
          tgt_tokens: (B, T)   目标词序列 (含 <bos>)
          v_s:        (B, F, visual_dim)
          v_a:        (B, F, visual_dim)
          C:          (B, K, concept_dim)
        Returns:
          logits:     (B, T, vocab_size)
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        # 投影到 d_model
        v_s = self.proj_vs(v_s)
        v_a = self.proj_va(v_a)
        C   = self.proj_C(C)

        # 词嵌入 + 位置编码
        w = self.word_embed(tgt_tokens) * math.sqrt(self.d_model)
        w = self.pos_enc(w)
        w = self.dropout(w)

        # causal mask
        tgt_mask = self.generate_causal_mask(T, device)

        for layer in self.layers:
            w = layer(w, C, v_s, v_a, tgt_mask=tgt_mask)

        logits = self.out_proj(w)  # (B, T, V)
        return logits

    @torch.no_grad()
    def greedy_decode(self,
                      v_s: torch.Tensor,
                      v_a: torch.Tensor,
                      C: torch.Tensor,
                      bos_id: int,
                      eos_id: int,
                      max_len: int = None) -> torch.Tensor:
        """简单贪心解码 (推理用)"""
        if max_len is None:
            max_len = self.max_len

        B = v_s.size(0)
        device = v_s.device
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(ys, v_s, v_a, C)        # (B, T, V)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys

    @torch.no_grad()
    def beam_search(self,
                    v_s: torch.Tensor,
                    v_a: torch.Tensor,
                    C: torch.Tensor,
                    bos_id: int,
                    eos_id: int,
                    beam_size: int = 6,
                    max_len: int = None) -> torch.Tensor:
        """
        Beam search 解码 (论文中 beam_size=6)，按 batch 内逐样本进行。
        返回每个样本最佳的 token 序列 list。
        """
        if max_len is None:
            max_len = self.max_len

        B = v_s.size(0)
        device = v_s.device
        results = []

        for b in range(B):
            v_s_b = v_s[b:b + 1]
            v_a_b = v_a[b:b + 1]
            C_b = C[b:b + 1]

            # 初始 beam: (seq, log_prob)
            beams = [(torch.tensor([[bos_id]], dtype=torch.long, device=device), 0.0, False)]

            for _ in range(max_len - 1):
                new_beams = []
                for seq, score, finished in beams:
                    if finished:
                        new_beams.append((seq, score, True))
                        continue
                    logits = self.forward(seq, v_s_b, v_a_b, C_b)  # (1,T,V)
                    log_probs = F.log_softmax(logits[:, -1], dim=-1).squeeze(0)
                    # beam_size 不能超过词表大小
                    k = min(beam_size, log_probs.size(-1))
                    topk_lp, topk_idx = log_probs.topk(k)
                    for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                        new_seq = torch.cat(
                            [seq, torch.tensor([[idx]], dtype=torch.long, device=device)],
                            dim=1)
                        new_beams.append((new_seq, score + lp, idx == eos_id))
                # 取 top beam_size
                new_beams.sort(key=lambda x: x[1] / max(1, x[0].size(1)),
                               reverse=True)
                beams = new_beams[:beam_size]

                if all(b[2] for b in beams):
                    break

            best_seq = beams[0][0].squeeze(0)
            results.append(best_seq)

        return results


# -----------------------------
# 长度加权交叉熵损失 (公式 16)
# -----------------------------
class LengthWeightedCELoss(nn.Module):
    """
    公式 (16) + Label Smoothing:
      L = -(1/N) Σ_i (1/γ_i^β) Σ_t CE_smooth(ŷ_{i,t}, y_{i,t})
    """

    def __init__(self, pad_id: int = 0, beta: float = 0.4,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.pad_id = pad_id
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logits:  (B, T, V)
          targets: (B, T)
        """
        B, T, V = logits.shape

        # 1) 每个样本的有效长度 γ_i (不计 pad)
        non_pad = (targets != self.pad_id).float()                 # (B,T)
        gamma_i = non_pad.sum(dim=1).clamp(min=1.0)                # (B,)

        # 2) per-token CE with label smoothing
        log_probs = F.log_softmax(logits, dim=-1)                  # (B,T,V)
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B,T)

        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=-1)                  # (B,T)
            ls = self.label_smoothing
            per_token = (1 - ls) * nll + ls * smooth_loss
        else:
            per_token = nll

        per_token = per_token * non_pad                            # mask pad

        # 3) 样本级求和
        per_sample = per_token.sum(dim=1)                          # (B,)

        # 4) 样本级长度加权: 1 / γ_i^β
        weight = 1.0 / (gamma_i.pow(self.beta))
        loss = (per_sample * weight).mean()
        return loss


if __name__ == "__main__":
    B, T, F_ = 2, 10, 26
    V, K = 1000, 300
    D_v, D_c, D_m = 1024, 512, 512

    decoder = DescriptionGenerator(
        vocab_size=V, d_model=D_m, num_layers=6, num_heads=8,
        visual_dim=D_v, concept_dim=D_c, max_len=26,
    )

    tgt = torch.randint(1, V, (B, T))
    v_s = torch.randn(B, F_, D_v)
    v_a = torch.randn(B, F_, D_v)
    C = torch.randn(B, K, D_c)

    logits = decoder(tgt, v_s, v_a, C)
    print("logits:", logits.shape)  # (2,10,1000)

    loss_fn = LengthWeightedCELoss(pad_id=0, beta=0.4)
    loss = loss_fn(logits, tgt)
    print("loss:", loss.item())
"""
Semantic Concept Detection (SCD) Module
对应论文 Section 3.3

功能：
1. 从训练集词表半自动选取 K 个最频繁、最有意义的名词/动词/形容词作为候选标签
2. 用 MLP 学习从视频全局特征 v_t 到 K 维语义标签向量的映射
3. 输出 K 个语义概念的概率分布 q_{i,j}
4. 训练损失为多标签 BCE (公式 11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticConceptDetector(nn.Module):
    """
    Args:
      input_dim:    全局视频特征维度 (一般为 D_a + D_m 或 pooled 后的维度)
      num_concepts: K，候选语义概念数量 (论文: MSVD=300, MSR-VTT=400, TBD=300)
      embed_dim:    每个语义概念的 embedding 维度 (用于交叉注意力)
      hidden_dim:   MLP 隐藏层维度
    """

    def __init__(self,
                 input_dim: int = 6144,   # 2048 + 4096
                 num_concepts: int = 300,
                 embed_dim: int = 512,
                 hidden_dim: int = 1024,
                 dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        self.embed_dim = embed_dim

        # MLP: 全局视频特征 → K 维语义概率
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

        # K 个语义概念的可学习 embedding (类似 GloVe 初始化的概念词向量)
        # 论文中使用 GloVe (Pennington et al., 2014)
        self.concept_embeddings = nn.Embedding(num_concepts, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.concept_embeddings.weight)

    def load_glove_embeddings(self, glove_vectors: torch.Tensor):
        """可选：用预训练 GloVe 词向量初始化 concept embedding"""
        assert glove_vectors.shape == self.concept_embeddings.weight.shape, \
            f"Shape mismatch: {glove_vectors.shape} vs {self.concept_embeddings.weight.shape}"
        with torch.no_grad():
            self.concept_embeddings.weight.copy_(glove_vectors)

    def forward(self, global_feats: torch.Tensor):
        """
        Args:
          global_feats: (B, input_dim)  视频全局特征 (例如各帧 v_t 的均值)
        Returns:
          q:    (B, K)        每个概念的概率
          C:    (B, K, E)     加权后的概念 embedding 序列 (供 decoder 使用)
        """
        # 1) 预测语义概率
        logits = self.mlp(global_feats)        # (B, K)
        q = torch.sigmoid(logits)              # (B, K)

        # 2) 概念 embedding 用概率加权
        # concept_embeddings.weight: (K, E)
        C = self.concept_embeddings.weight.unsqueeze(0).expand(global_feats.size(0), -1, -1)  # (B,K,E)
        C = C * q.unsqueeze(-1)                # (B, K, E)

        return q, C

    @staticmethod
    def bce_loss(q: torch.Tensor, p_hat: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        公式 (11):
        L_S = -(1/N) Σ_i Σ_j [ p̂_{i,j} log q_{i,j} + (1 - p̂_{i,j}) log (1 - q_{i,j}) ]

        q:     (B, K)  预测概率
        p_hat: (B, K)  ground-truth 标签 (0/1)
        """
        q = q.clamp(min=eps, max=1.0 - eps)
        loss = -(p_hat * torch.log(q) + (1.0 - p_hat) * torch.log(1.0 - q))
        return loss.mean()


if __name__ == "__main__":
    B, D_in, K, E = 4, 6144, 300, 512
    feats = torch.randn(B, D_in)
    scd = SemanticConceptDetector(input_dim=D_in, num_concepts=K, embed_dim=E)
    q, C = scd(feats)
    print("q:", q.shape, " C:", C.shape)  # (4,300) (4,300,512)

    p_hat = (torch.rand(B, K) > 0.7).float()
    loss = scd.bce_loss(q, p_hat)
    print("BCE loss:", loss.item())
"""
Evaluation Metrics
对应论文 Section 4.2:
  - BLEU_1/2/3/4 (Papineni et al. 2002)
  - METEOR (Denkowski & Lavie 2014)
  - ROUGE_L (Lin 2004)
  - CIDEr (Vedantam et al. 2015)

依赖: pycocoevalcap (推荐) 或自带的简化实现
"""

import math
from collections import Counter
from typing import Dict, List


# -----------------------------------------------------------
# 自带的纯 Python BLEU (作为 fallback)
# -----------------------------------------------------------


def _tokenize_for_eval(text: str) -> list:
    """
    评估专用分词：与 vocab.py 的 tokenize_caption 保持一致。
    中文 → 字符级；英文/数字 → 空格切分。
    """
    tokens = []
    buf = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            if buf:
                tokens.extend(''.join(buf).lower().split())
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        elif ch.isspace():
            if buf:
                tokens.extend(''.join(buf).lower().split())
                buf = []
        else:
            if buf:
                tokens.extend(''.join(buf).lower().split())
                buf = []
    if buf:
        tokens.extend(''.join(buf).lower().split())
    return [t for t in tokens if t]

def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(hyp: List[str], refs: List[List[str]], n: int) -> tuple:
    hyp_ngrams = Counter(_ngrams(hyp, n))
    if not hyp_ngrams:
        return 0, 0
    max_ref = Counter()
    for ref in refs:
        ref_ng = Counter(_ngrams(ref, n))
        for k, v in ref_ng.items():
            max_ref[k] = max(max_ref[k], v)
    clipped = {k: min(v, max_ref[k]) for k, v in hyp_ngrams.items()}
    return sum(clipped.values()), sum(hyp_ngrams.values())


def _brevity_penalty(hyp_len: int, ref_lens: List[int]) -> float:
    closest = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    if hyp_len > closest:
        return 1.0
    if hyp_len == 0:
        return 0.0
    return math.exp(1.0 - closest / hyp_len)


def simple_bleu(hypotheses: List[str],
                references: List[List[str]],
                max_n: int = 4) -> Dict[str, float]:
    """
    Args:
      hypotheses: list of str
      references: list of list of str  (每个样本的多句 reference)
    """
    correct = [0] * max_n
    total = [0] * max_n
    hyp_lens, ref_lens_all = 0, []

    for hyp, refs in zip(hypotheses, references):
        hyp_tok = _tokenize_for_eval(hyp)
        refs_tok = [_tokenize_for_eval(r) for r in refs]
        hyp_lens += len(hyp_tok)
        ref_lens_all.append(min((len(r) for r in refs_tok),
                                key=lambda x: abs(x - len(hyp_tok))))
        for n in range(1, max_n + 1):
            c, t = _modified_precision(hyp_tok, refs_tok, n)
            correct[n - 1] += c
            total[n - 1] += t

    precisions = [(c / t) if t > 0 else 0.0 for c, t in zip(correct, total)]
    bp = _brevity_penalty(hyp_lens, ref_lens_all) if ref_lens_all else 0.0

    bleus = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p = precisions[n - 1]
        log_sum += math.log(p) if p > 0 else -1e9
        bleus[f'BLEU_{n}'] = bp * math.exp(log_sum / n) * 100
    return bleus


# -----------------------------------------------------------
# 简化 ROUGE-L (基于 LCS)
# -----------------------------------------------------------
def _lcs(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def simple_rouge_l(hypotheses, references, beta: float = 1.2) -> float:
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tok = _tokenize_for_eval(hyp)
        best = 0.0
        for ref in refs:
            ref_tok = _tokenize_for_eval(ref)
            lcs = _lcs(hyp_tok, ref_tok)
            if lcs == 0:
                continue
            p = lcs / len(hyp_tok) if hyp_tok else 0
            r = lcs / len(ref_tok) if ref_tok else 0
            if p + r == 0:
                continue
            f = ((1 + beta**2) * p * r) / (r + beta**2 * p)
            best = max(best, f)
        scores.append(best)
    return (sum(scores) / len(scores) * 100) if scores else 0.0


# -----------------------------------------------------------
# 简化 METEOR (unigram F-score with α=0.9, 不含同义词扩展)
# -----------------------------------------------------------
def simple_meteor(hypotheses, references, alpha: float = 0.9) -> float:
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tok = _tokenize_for_eval(hyp)
        best = 0.0
        for ref in refs:
            ref_tok = _tokenize_for_eval(ref)
            matches = sum((Counter(hyp_tok) & Counter(ref_tok)).values())
            if matches == 0:
                continue
            p = matches / len(hyp_tok) if hyp_tok else 0
            r = matches / len(ref_tok) if ref_tok else 0
            if p + r == 0:
                continue
            f = (p * r) / (alpha * p + (1 - alpha) * r)
            best = max(best, f)
        scores.append(best)
    return (sum(scores) / len(scores) * 100) if scores else 0.0


# -----------------------------------------------------------
# 简化 CIDEr (基于 TF-IDF n-gram 余弦相似)
# -----------------------------------------------------------
def simple_cider(hypotheses, references, n: int = 4, sigma: float = 6.0) -> float:
    # 计算每个 n-gram 在所有 references 中的文档频率
    doc_freq = [Counter() for _ in range(n)]
    num_docs = 0
    for refs in references:
        num_docs += 1
        seen = [set() for _ in range(n)]
        for ref in refs:
            ref_tok = _tokenize_for_eval(ref)
            for k in range(n):
                for ng in _ngrams(ref_tok, k + 1):
                    seen[k].add(ng)
        for k in range(n):
            for ng in seen[k]:
                doc_freq[k][ng] += 1

    def vec(tokens, k):
        ng = Counter(_ngrams(tokens, k + 1))
        v = {}
        norm = 0.0
        for g, c in ng.items():
            df = doc_freq[k].get(g, 0)
            if df == 0:
                idf = 0
            else:
                idf = math.log(max(num_docs, 1) / df)
            w = c * idf
            v[g] = w
            norm += w * w
        return v, math.sqrt(norm)

    def cosine(v1, n1, v2, n2):
        if n1 == 0 or n2 == 0:
            return 0.0
        s = sum(v1[g] * v2.get(g, 0) for g in v1)
        return s / (n1 * n2)

    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tok = _tokenize_for_eval(hyp)
        per_n = []
        for k in range(n):
            hv, hn = vec(hyp_tok, k)
            ref_scores = []
            for ref in refs:
                rv, rn = vec(ref.split(), k)
                cos = cosine(hv, hn, rv, rn)
                # length penalty
                delta = len(hyp_tok) - len(ref.split())
                cos *= math.exp(-(delta ** 2) / (2 * sigma * sigma))
                ref_scores.append(cos)
            per_n.append(10 * (sum(ref_scores) / len(ref_scores)) if ref_scores else 0)
        scores.append(sum(per_n) / n)
    return (sum(scores) / len(scores)) if scores else 0.0


# -----------------------------------------------------------
# 综合评估
# -----------------------------------------------------------
def evaluate_captions(hypotheses: Dict[str, str],
                      references: Dict[str, List[str]],
                      use_pycoco: bool = True) -> Dict[str, float]:
    """
    Args:
      hypotheses: {video_id: 'generated caption'}
      references: {video_id: ['ref1', 'ref2', ...]}
    Returns:
      {BLEU_1, BLEU_2, BLEU_3, BLEU_4, METEOR, ROUGE_L, CIDEr}
    """
    common = sorted(set(hypotheses) & set(references))
    hyps = [hypotheses[k].strip() for k in common]
    refs = [[r.strip() for r in references[k]] for k in common]

    if use_pycoco:
        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider

            gts = {k: refs[i] for i, k in enumerate(common)}
            res = {k: [hyps[i]] for i, k in enumerate(common)}

            scores = {}
            bleu_score, _ = Bleu(4).compute_score(gts, res)
            for i, s in enumerate(bleu_score):
                scores[f'BLEU_{i+1}'] = s * 100

            meteor, _ = Meteor().compute_score(gts, res)
            scores['METEOR'] = meteor * 100

            rouge, _ = Rouge().compute_score(gts, res)
            scores['ROUGE_L'] = rouge * 100

            cider, _ = Cider().compute_score(gts, res)
            scores['CIDEr'] = cider * 100

            return scores
        except Exception as e:
            print(f"[evaluate_captions] pycocoevalcap failed ({e}), fallback to simple impl.")

    # Fallback
    scores = simple_bleu(hyps, refs)
    scores['METEOR']  = simple_meteor(hyps, refs)
    scores['ROUGE_L'] = simple_rouge_l(hyps, refs)
    scores['CIDEr']   = simple_cider(hyps, refs)
    return scores


if __name__ == "__main__":
    hyps = {
        'v1': 'the teacher stands in front of the classroom',
        'v2': 'a teacher writes on the whiteboard',
    }
    refs = {
        'v1': ['the male teacher stands at the front of the classroom',
               'a teacher in front of the classroom'],
        'v2': ['the teacher is writing on the whiteboard',
               'teacher writes content on a whiteboard'],
    }
    print(evaluate_captions(hyps, refs, use_pycoco=False))
"""
Vocabulary & Caption Preprocessing
对应论文 Section 4.2:
  - 所有 reference caption 转小写并去标点
  - TBD 词表 ~3998 tokens, 词频 ≥2 入表
  - caption pad/trim 到 26 词
  - 起止符 <bos>/<eos>, OOV → <unk>
  - 词嵌入 512 维 (one-hot → Embedding 学习)

另外:
  - SCD 候选语义概念: MSVD 300 / MSR-VTT 400 / TBD 300,
    半自动选取训练集中最频繁的名词/动词/形容词
"""

import json
import re
import string
from collections import Counter
from typing import List, Dict


# 特殊 token
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'
UNK = '<unk>'
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize_caption(text: str) -> List[str]:
    """
    多语言 tokenizer：
    - 英文：小写化 + 去标点 + 空格切分
    - 中文：字符级切分（每个汉字作为一个 token）
    - 中英混合：先字符级处理中文，再空格切分英文
    """
    import unicodedata

    tokens = []
    buf = []   # 缓存连续英文/数字字符

    for ch in text:
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, '')

        # 中文字符（CJK 统一汉字范围）
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            if buf:
                # 先把缓存的英文刷出来
                word = ''.join(buf).lower().strip()
                if word:
                    tokens.extend(word.split())
                buf = []
            tokens.append(ch)

        # 字母或数字 → 缓存
        elif ch.isalnum():
            buf.append(ch)

        # 空格 → 刷缓存
        elif ch.isspace():
            if buf:
                word = ''.join(buf).lower().strip()
                if word:
                    tokens.extend(word.split())
                buf = []

        # 标点/其他 → 丢弃，但先刷缓存
        else:
            if buf:
                word = ''.join(buf).lower().strip()
                if word:
                    tokens.extend(word.split())
                buf = []

    # 处理末尾缓存
    if buf:
        word = ''.join(buf).lower().strip()
        if word:
            tokens.extend(word.split())

    return [t for t in tokens if t]


class Vocabulary:
    """简易词表"""

    def __init__(self,
                 min_freq: int = 2,
                 max_size: int = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

    @property
    def pad_id(self): return self.word2idx[PAD]
    @property
    def bos_id(self): return self.word2idx[BOS]
    @property
    def eos_id(self): return self.word2idx[EOS]
    @property
    def unk_id(self): return self.word2idx[UNK]

    def __len__(self): return len(self.word2idx)

    def build(self, all_captions: List[str]):
        """从全部 caption 文本构建词表"""
        counter = Counter()
        for cap in all_captions:
            counter.update(tokenize_caption(cap))

        # 先放特殊符号
        for tok in SPECIALS:
            self._add(tok)

        # 按频次降序
        sorted_words = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for word, freq in sorted_words:
            if freq < self.min_freq:
                continue
            if self.max_size and len(self) >= self.max_size:
                break
            self._add(word)

        return counter  # 返回词频供 SCD 选标签使用

    def _add(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 26,
               add_special: bool = True) -> List[int]:
        """
        把文本编码成定长 token ids:
          [<bos>, w1, ..., wk, <eos>, <pad>, ...]
        若不加特殊符号, 直接 trim/pad。
        """
        toks = tokenize_caption(text)
        ids = [self.word2idx.get(t, self.unk_id) for t in toks]

        if add_special:
            ids = [self.bos_id] + ids[:max_len - 2] + [self.eos_id]
        else:
            ids = ids[:max_len]

        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK)
            if strip_special and w in (PAD, BOS, EOS):
                if w == EOS:
                    break
                continue
            words.append(w)
        return ' '.join(words)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        vocab = cls()
        with open(path, 'r', encoding='utf-8') as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


# -----------------------------------------------------------
# SCD 语义概念词表 (从训练集词频中半自动选取)
# -----------------------------------------------------------
def build_concept_vocab(all_captions: List[str],
                        num_concepts: int,
                        pos_filter: bool = True) -> List[str]:
    """
    论文: "These k semantic concepts are selected semi-manually from the
    training set's vocabulary, focusing on the most frequent and
    meaningful nouns, verbs, and adjectives."

    Args:
      num_concepts: K
      pos_filter:   是否用 nltk POS-tag 过滤为 n/v/adj
    """
    counter = Counter()
    for cap in all_captions:
        counter.update(tokenize_caption(cap))

    # 去除 stop words
    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'and', 'or', 'but', 'if', 'while', 'with', 'at', 'by', 'for', 'of',
        'in', 'on', 'to', 'from', 'this', 'that', 'these', 'those', 'it',
        'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'i',
        'we', 'you', 'us', 'me', 'my', 'your', 'who', 'which', 'what',
        'how', 'why', 'so', 'such', 'than', 'then', 'as', 'do', 'does',
        'did', 'have', 'has', 'had',
    }

    if pos_filter:
        try:
            import nltk
            nltk.download('averaged_perceptron_tagger', quiet=True)
            words = [w for w in counter if w not in stopwords]
            tagged = nltk.pos_tag(words)
            keep = {w for w, tag in tagged
                    if tag.startswith(('NN', 'VB', 'JJ'))}
            filtered = [(w, c) for w, c in counter.items()
                        if w in keep and w not in stopwords]
        except Exception:
            filtered = [(w, c) for w, c in counter.items()
                        if w not in stopwords]
    else:
        filtered = [(w, c) for w, c in counter.items() if w not in stopwords]

    filtered.sort(key=lambda x: (-x[1], x[0]))
    return [w for w, _ in filtered[:num_concepts]]


def captions_to_concept_labels(captions: List[str],
                               concept_vocab: List[str]) -> List[int]:
    """
    公式 (11) 对应的 ground-truth p̂_{i,j}:
      若第 j 个概念出现在至少一个 caption 中 → 1, 否则 0。
    """
    concept_set = set(concept_vocab)
    concept2idx = {w: i for i, w in enumerate(concept_vocab)}
    label = [0] * len(concept_vocab)
    for cap in captions:
        for tok in tokenize_caption(cap):
            if tok in concept_set:
                label[concept2idx[tok]] = 1
    return label


if __name__ == "__main__":
    sample_caps = [
        "The teacher stands at the front of the classroom, writing on the whiteboard.",
        "A teacher walks around the classroom, engaging with students.",
        "The teacher points to a diagram on the blackboard.",
    ]
    vocab = Vocabulary(min_freq=1)
    counter = vocab.build(sample_caps)
    print("vocab size:", len(vocab))

    ids = vocab.encode(sample_caps[0], max_len=20)
    print("encoded:", ids)
    print("decoded:", vocab.decode(ids))

    concepts = build_concept_vocab(sample_caps, num_concepts=10, pos_filter=False)
    print("concepts:", concepts)
    labels = captions_to_concept_labels(sample_caps[:1], concepts)
    print("labels:", labels)
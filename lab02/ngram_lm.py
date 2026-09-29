"""Cài đặt từ đầu mô hình ngôn ngữ unigram, bigram và trigram."""

from __future__ import annotations

from collections import Counter
from math import exp, log
import re
from typing import Iterable, Sequence

TokenizedCorpus = Iterable[Sequence[str]]
NGram = tuple[str, ...]
UNK = "<unk>"
START = "<s>"
END = "</s>"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def tokenize(text: str) -> list[str]:
    return [match.group().lower().replace("’", "'")
            for match in TOKEN_PATTERN.finditer(text)]


def text_to_sentences(text: str) -> list[list[str]]:
    return [tokens for part in SENTENCE_PATTERN.split(text)
            if (tokens := tokenize(part))]


def build_vocabulary(corpus: TokenizedCorpus, min_count: int = 1) -> set[str]:
    """Tạo vocabulary từ các token đạt ``min_count``, kèm token chưa biết."""
    counts = Counter(token for sentence in corpus for token in sentence)
    return {token for token, count in counts.items() if count >= min_count} | {UNK, END}


def count_ngrams(corpus: TokenizedCorpus, n: int) -> Counter[NGram]:
    """Đếm n-gram trong các câu đã token hóa, có thêm ký hiệu đầu/cuối câu."""
    if n < 1:
        raise ValueError("n phải từ 1 trở lên")
    counts: Counter[NGram] = Counter()
    for sentence in corpus:
        padded = [START] * (n - 1) + list(sentence) + [END]
        counts.update(tuple(padded[i : i + n]) for i in range(len(padded) - n + 1))
    return counts


class NGramLanguageModel:
    """Mô hình MLE hoặc Laplace, chỉ dùng tối đa ``n - 1`` từ làm ngữ cảnh."""

    def __init__(self, n: int, smoothing: str = "mle", alpha: float = 1.0,
                 min_count: int = 1):
        if n not in (1, 2, 3):
            raise ValueError("Bài lab chỉ hỗ trợ unigram, bigram và trigram")
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing phải là 'mle' hoặc 'laplace'")
        if alpha <= 0:
            raise ValueError("alpha phải lớn hơn 0")
        if min_count < 1:
            raise ValueError("min_count phải từ 1 trở lên")
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        self.min_count = min_count
        self.vocabulary: set[str] = set()
        self.ngram_counts: Counter[NGram] = Counter()
        self.context_counts: Counter[NGram] = Counter()
        self.fitted = False

    def fit(self, corpus: TokenizedCorpus) -> "NGramLanguageModel":
        sentences = [list(sentence) for sentence in corpus if sentence]
        self.vocabulary = build_vocabulary(sentences, self.min_count)
        mapped = [[token if token in self.vocabulary else UNK for token in s]
                  for s in sentences]
        self.ngram_counts = count_ngrams(mapped, self.n)
        self.context_counts = Counter()
        for gram, count in self.ngram_counts.items():
            self.context_counts[gram[:-1]] += count
        self.fitted = True
        return self

    def with_smoothing(self, smoothing: str, alpha: float = 1.0) -> "NGramLanguageModel":
        """Tạo biến thể smoothing dùng chung các count đã học, không sao chép Counter lớn."""
        if not self.fitted:
            raise RuntimeError("Cần gọi fit() trước khi tạo biến thể smoothing")
        variant = NGramLanguageModel(self.n, smoothing=smoothing, alpha=alpha,
                                     min_count=self.min_count)
        variant.vocabulary = self.vocabulary
        variant.ngram_counts = self.ngram_counts
        variant.context_counts = self.context_counts
        variant.fitted = True
        return variant

    def _context(self, context: Sequence[str]) -> NGram:
        if self.n == 1:
            return ()
        padded = [START] * max(0, self.n - 1 - len(context)) + list(context)
        return tuple(token if token in self.vocabulary or token == START else UNK
                     for token in padded[-(self.n - 1):])

    def probability(self, context: Sequence[str] | str, word: str) -> float:
        if not self.fitted:
            raise RuntimeError("Cần gọi fit() trước probability()")
        if isinstance(context, str):
            context = tokenize(context)
        if word in (UNK, END):
            mapped_word = word
        else:
            normalized_word = tokenize(word)
            mapped_word = normalized_word[0] if len(normalized_word) == 1 else UNK
            if mapped_word not in self.vocabulary:
                mapped_word = UNK
        history = self._context(context)
        count = self.ngram_counts[history + (mapped_word,)]
        denominator = self.context_counts[history]
        if self.smoothing == "laplace":
            return (count + self.alpha) / (denominator + self.alpha * len(self.vocabulary))
        return count / denominator if denominator else 0.0

    def sentence_log_probability(self, sentence: Sequence[str] | str) -> float:
        if isinstance(sentence, str):
            sentence = tokenize(sentence)
        tokens = [token if token in self.vocabulary else UNK for token in sentence]
        history = [START] * (self.n - 1)
        score = 0.0
        for word in [*tokens, END]:
            probability = self.probability(history, word)
            if probability == 0.0:
                return float("-inf")
            score += log(probability)
            history.append(word)
            history = history[-(self.n - 1):] if self.n > 1 else []
        return score

    def sentence_probability(self, sentence: Sequence[str] | str) -> float:
        score = self.sentence_log_probability(sentence)
        return exp(score) if score > -745 else 0.0

    def next_word_distribution(self, context: Sequence[str] | str) -> dict[str, float]:
        if isinstance(context, str):
            context = tokenize(context)
        history = self._context(context)
        if self.smoothing == "laplace":
            candidates = self.vocabulary
        else:
            candidates = {gram[-1] for gram in self.ngram_counts
                         if gram[:-1] == history}
        return {word: self.probability(history, word)
                for word in sorted(candidates)}

    def predict_next(self, context: Sequence[str] | str, top_k: int = 5) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k phải từ 1 trở lên")
        distribution = self.next_word_distribution(context)
        return sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def continuation_log_probability(self, context: Sequence[str] | str,
                                     continuation: Sequence[str] | str) -> float:
        if isinstance(context, str):
            context = tokenize(context)
        if isinstance(continuation, str):
            continuation = tokenize(continuation)
        history = list(context)
        score = 0.0
        for word in continuation:
            probability = self.probability(history, word)
            if probability == 0.0:
                return float("-inf")
            score += log(probability)
            history.append(word)
        return score

    def perplexity(self, corpus: TokenizedCorpus) -> float:
        total_log_probability = 0.0
        predicted_tokens = 0
        for sentence in corpus:
            if not sentence:
                continue
            score = self.sentence_log_probability(sentence)
            if score == float("-inf"):
                return float("inf")
            total_log_probability += score
            predicted_tokens += len(sentence) + 1  # Tính cả </s> nhất quán.
        return exp(-total_log_probability / predicted_tokens) if predicted_tokens else float("inf")


def train_unigram(corpus: TokenizedCorpus, **kwargs) -> NGramLanguageModel:
    return NGramLanguageModel(1, **kwargs).fit(corpus)


def train_bigram(corpus: TokenizedCorpus, **kwargs) -> NGramLanguageModel:
    return NGramLanguageModel(2, **kwargs).fit(corpus)


def train_trigram(corpus: TokenizedCorpus, **kwargs) -> NGramLanguageModel:
    return NGramLanguageModel(3, **kwargs).fit(corpus)


def frequency_summary(corpus: TokenizedCorpus, n: int) -> dict[str, int]:
    counts = count_ngrams(corpus, n)
    return {
        "unique_ngrams": len(counts),
        "singleton_ngrams": sum(count == 1 for count in counts.values()),
        "total_ngrams": sum(counts.values()),
    }

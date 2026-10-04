"""Hàm cho ma trận co-occurrence và các phép so sánh word vector."""

from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import islice

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse


def build_vocabulary(
    corpus: Iterable[Sequence[str]],
    min_count: int = 1,
    max_size: int | None = None,
) -> list[str]:
    """Tạo vocabulary theo tần suất giảm dần, hòa thì sắp xếp alphabet."""
    if min_count < 1:
        raise ValueError("min_count phải từ 1 trở lên")
    if max_size is not None and max_size < 1:
        raise ValueError("max_size phải từ 1 trở lên")

    counts = Counter(token for sentence in corpus for token in sentence)
    words = [word for word, count in counts.items() if count >= min_count]
    words.sort(key=lambda word: (-counts[word], word))
    return words if max_size is None else words[:max_size]


def build_cooccurrence_matrix(
    corpus: Iterable[Sequence[str]],
    vocabulary: Sequence[str],
    window_size: int,
) -> csr_matrix:
    """Tạo ma trận sparse target × context, đếm cả hai phía của cửa sổ."""
    if window_size < 1:
        raise ValueError("window_size phải từ 1 trở lên")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("vocabulary không được chứa từ trùng lặp")

    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    size = len(vocabulary)
    matrix = csr_matrix((size, size), dtype=np.int32)
    iterator = iter(corpus)
    chunk_size = 512
    while chunk := list(islice(iterator, chunk_size)):
        row_indices: list[int] = []
        column_indices: list[int] = []
        counts: list[int] = []
        for sentence in chunk:
            indexed_sentence = [word_to_index.get(word, -1) for word in sentence]
            for target_position, target_index in enumerate(indexed_sentence):
                if target_index < 0:
                    continue
                start = max(0, target_position - window_size)
                end = min(len(indexed_sentence), target_position + window_size + 1)
                for context_position in range(start, end):
                    if context_position == target_position:
                        continue
                    context_index = indexed_sentence[context_position]
                    if context_index < 0:
                        continue
                    row_indices.append(target_index)
                    column_indices.append(context_index)
                    counts.append(1)

        chunk_matrix = coo_matrix(
            (np.asarray(counts, dtype=np.int32), (row_indices, column_indices)),
            shape=(size, size),
            dtype=np.int32,
        ).tocsr()
        chunk_matrix.sum_duplicates()
        matrix = matrix + chunk_matrix
    return matrix.tocsr()


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Tính cosine similarity cho dense hoặc scipy sparse vector."""
    if issparse(vector_a):
        vector_a = vector_a.toarray()
    if issparse(vector_b):
        vector_b = vector_b.toarray()
    a = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    b = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Hai vector phải có cùng số chiều")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def most_similar(
    word: str,
    matrix: csr_matrix,
    vocabulary: Sequence[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Trả về top-k từ gần word nhất, không tính chính từ đó."""
    if top_k < 1:
        raise ValueError("top_k phải từ 1 trở lên")
    if matrix.shape != (len(vocabulary), len(vocabulary)):
        raise ValueError("Kích thước matrix không khớp với vocabulary")
    try:
        target_index = vocabulary.index(word)
    except ValueError as error:
        raise KeyError(f"Từ {word!r} không có trong vocabulary") from error

    target_vector = matrix.getrow(target_index)
    if target_vector.nnz == 0:
        return []

    dot_products = (matrix @ target_vector.transpose()).toarray().reshape(-1)
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1))
    denominator = row_norms * np.linalg.norm(target_vector.data)
    scores = np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products, dtype=np.float64),
        where=denominator != 0,
    )
    ranked_indices = sorted(
        (index for index in range(len(vocabulary)) if index != target_index),
        key=lambda index: (-scores[index], vocabulary[index]),
    )
    return [(vocabulary[index], float(scores[index]))
            for index in ranked_indices[:top_k]]


def build_cbow_examples(
    sentence: Sequence[str],
    window_size: int = 1,
) -> list[tuple[list[str], str]]:
    """Tạo ví dụ (context, target) cho CBOW từ một câu."""
    if window_size < 1:
        raise ValueError("window_size phải từ 1 trở lên")
    examples = []
    for target_position, target in enumerate(sentence):
        start = max(0, target_position - window_size)
        end = min(len(sentence), target_position + window_size + 1)
        context = [sentence[i] for i in range(start, end) if i != target_position]
        if context:
            examples.append((context, target))
    return examples


def build_skipgram_pairs(
    sentence: Sequence[str],
    window_size: int = 1,
) -> list[tuple[str, str]]:
    """Tạo các cặp (target, context) cho Skip-gram từ một câu."""
    if window_size < 1:
        raise ValueError("window_size phải từ 1 trở lên")
    pairs = []
    for target_position, target in enumerate(sentence):
        start = max(0, target_position - window_size)
        end = min(len(sentence), target_position + window_size + 1)
        pairs.extend(
            (target, sentence[context_position])
            for context_position in range(start, end)
            if context_position != target_position
        )
    return pairs


def context_evidence(
    corpus: Iterable[Sequence[str]],
    word: str,
    window_size: int = 2,
    top_k: int = 10,
) -> list[tuple[str, int]]:
    """Đếm các từ thường xuất hiện cạnh word để dùng làm bằng chứng ngữ cảnh."""
    if window_size < 1:
        raise ValueError("window_size phải từ 1 trở lên")
    if top_k < 1:
        raise ValueError("top_k phải từ 1 trở lên")

    counts: Counter[str] = Counter()
    for sentence in corpus:
        for position, token in enumerate(sentence):
            if token != word:
                continue
            start = max(0, position - window_size)
            end = min(len(sentence), position + window_size + 1)
            counts.update(
                sentence[context_position]
                for context_position in range(start, end)
                if context_position != position
            )
    return counts.most_common(top_k)

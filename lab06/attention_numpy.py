"""Các phép tính attention cơ bản cài bằng NumPy cho LAB 06."""

import numpy as np


def stable_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính softmax ổn định, trừ max trước khi lấy exponential."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0:
        raise ValueError("scores cần ít nhất một chiều")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores chỉ được chứa giá trị hữu hạn")

    shifted_scores = scores - np.max(scores, axis=axis, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    return exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    temperature: float = 1.0,
):
    """Tính scores, weights và context cho query/key/value.

    Hỗ trợ vector query (d_k,), ma trận query (T_q, d_k) hoặc batch có
    leading dimensions tương thích để NumPy broadcast.
    """
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if query.ndim < 1 or keys.ndim < 2 or values.ndim < 2:
        raise ValueError("query cần ít nhất 1D; keys và values cần ít nhất 2D")
    if query.shape[-1] != keys.shape[-1]:
        raise ValueError("query và keys phải có cùng key dimension")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("keys và values phải có cùng số token")
    if temperature <= 0:
        raise ValueError("temperature phải lớn hơn 0")
    if not all(np.all(np.isfinite(array)) for array in (query, keys, values)):
        raise ValueError("query, keys và values chỉ được chứa số hữu hạn")

    key_dimension = keys.shape[-1]
    scores = np.matmul(query, np.swapaxes(keys, -1, -2))
    scores = scores / np.sqrt(key_dimension)
    scores = scores / temperature
    attention_weights = stable_softmax(scores, axis=-1)
    context = np.matmul(attention_weights, values)
    return scores, attention_weights, context


def self_attention(
    sequence: np.ndarray,
    w_query: np.ndarray,
    w_key: np.ndarray,
    w_value: np.ndarray,
):
    """Tạo Q/K/V từ cùng một sequence và trả về các tensors trung gian.

    Trả về theo thứ tự: query, keys, values, scores, weights, context.
    """
    sequence = np.asarray(sequence, dtype=np.float64)
    w_query = np.asarray(w_query, dtype=np.float64)
    w_key = np.asarray(w_key, dtype=np.float64)
    w_value = np.asarray(w_value, dtype=np.float64)

    if sequence.ndim < 2:
        raise ValueError("sequence cần shape (..., số token, embedding dim)")
    if any(matrix.ndim != 2 for matrix in (w_query, w_key, w_value)):
        raise ValueError("W_Q, W_K và W_V cần là ma trận 2D")
    if any(matrix.shape[0] != sequence.shape[-1] for matrix in (w_query, w_key, w_value)):
        raise ValueError("Số hàng của từng W phải bằng embedding dimension")
    if w_query.shape[1] != w_key.shape[1]:
        raise ValueError("W_Q và W_K phải tạo cùng key dimension")

    query = np.matmul(sequence, w_query)
    keys = np.matmul(sequence, w_key)
    values = np.matmul(sequence, w_value)
    scores, weights, context = scaled_dot_product_attention(
        query, keys, values
    )
    return query, keys, values, scores, weights, context

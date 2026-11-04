"""Khung sinusoidal positional encoding cho LAB 07."""

import numpy as np


def sinusoidal_positional_encoding(
    max_length: int,
    d_model: int,
) -> np.ndarray:
    """Trả về PE có shape (max_length, d_model)."""
    if max_length <= 0 or d_model <= 0:
        raise ValueError("max_length và d_model phải lớn hơn 0")

    positions = np.arange(max_length, dtype=np.float64)[:, None]
    even_dimensions = np.arange(0, d_model, 2, dtype=np.float64)
    angle_rates = np.exp(-np.log(10000.0) * even_dimensions / d_model)
    angles = positions * angle_rates[None, :]

    encoding = np.zeros((max_length, d_model), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles[:, :encoding[:, 1::2].shape[1]])
    return encoding

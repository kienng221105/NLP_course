"""Cài đặt một Simple RNN cell bằng NumPy để quan sát một bước tính."""

import numpy as np


class SimpleRNNCell:
    """Một timestep: h_t = tanh(W_xh x_t + W_hh h_prev + b_h)."""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size và hidden_size phải lớn hơn 0")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)
        scale = 1 / np.sqrt(hidden_size)
        self.W_xh = self.rng.uniform(
            -scale, scale, size=(hidden_size, input_size)
        )
        self.W_hh = self.rng.uniform(
            -scale, scale, size=(hidden_size, hidden_size)
        )
        self.b_h = np.zeros(hidden_size, dtype=np.float64)

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Tính hidden state mới cho một timestep."""
        x_t = np.asarray(x_t, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)

        if x_t.shape != (self.input_size,):
            raise ValueError(
                f"x_t cần shape ({self.input_size},), nhận được {x_t.shape}"
            )
        if h_prev.shape != (self.hidden_size,):
            raise ValueError(
                f"h_prev cần shape ({self.hidden_size},), nhận được {h_prev.shape}"
            )

        a_t = self.W_xh @ x_t + self.W_hh @ h_prev + self.b_h
        return np.tanh(a_t)

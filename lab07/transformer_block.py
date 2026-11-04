"""Khung Transformer Encoder block tối giản cho LAB 07."""

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError("d_model và n_heads phải lớn hơn 0")
        if d_model % n_heads != 0:
            raise ValueError("d_model phải chia hết cho n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        """Input shape (B, T, d_model); padding_mask=True tại vị trí PAD."""
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x cần shape (batch, sequence_length, d_model)")

        batch_size, sequence_length, _ = x.shape

        def split_heads(projected):
            return projected.view(
                batch_size, sequence_length, self.n_heads, self.d_k
            ).transpose(1, 2)

        query = split_heads(self.query_projection(x))
        keys = split_heads(self.key_projection(x))
        values = split_heads(self.value_projection(x))

        scores = torch.matmul(query, keys.transpose(-2, -1))
        scores = scores / (self.d_k ** 0.5)

        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask cần shape (batch, sequence_length)")
            scores = scores.masked_fill(
                padding_mask[:, None, None, :].bool(),
                torch.finfo(scores.dtype).min,
            )

        attention_weights = torch.softmax(scores, dim=-1)
        dropped_weights = self.attention_dropout(attention_weights)
        context = torch.matmul(dropped_weights, values)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        output = self.output_dropout(self.output_projection(context))

        if return_attention:
            return output, attention_weights
        return output


class FeedForwardNetwork(nn.Module):
    """Position-wise FFN: Linear -> ReLU -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model và d_ff phải lớn hơn 0")
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = torch.relu(self.linear1(x))
        return self.linear2(self.dropout(hidden))


class TransformerBlock(nn.Module):
    """Một encoder block với hai residual connections và LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        attention_output, attention_weights = self.self_attention(
            x,
            padding_mask=padding_mask,
            return_attention=True,
        )
        x = self.attention_norm(x + self.residual_dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        output = self.feed_forward_norm(
            x + self.residual_dropout(feed_forward_output)
        )

        if return_attention:
            return output, attention_weights
        return output

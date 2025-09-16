from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from einops import rearrange, einsum

from cs336_basics.transformer.embedding import RotaryPositionEmbedding


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.amax(dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum


def cross_entropy(
    logits: Float[torch.Tensor, "batch vocab"],
    target_ids: Int[torch.Tensor, " batch"],
) -> Float[torch.Tensor, ""]:

    logits_max = logits.amax(dim=-1, keepdim=True)
    logits_exp = torch.exp(logits - logits_max)
    logits_exp_sum = logits_exp.sum(dim=-1, keepdim=True)

    log_probs = logits - logits_max - torch.log(logits_exp_sum)

    target_log_probs = log_probs[torch.arange(logits.shape[0]), target_ids]

    negative_log_likelihood = -target_log_probs.mean()
    return negative_log_likelihood


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch ... queries d_k"],
    K: Float[torch.Tensor, "batch ... keys d_k"],
    V: Float[torch.Tensor, "batch ... keys d_v"],
    mask: Bool[torch.Tensor, "batch ... queries keys"] | None = None,
) -> Float[torch.Tensor, "batch ... queries d_v"]:

    d_k = Q.shape[-1]

    QK = einsum(
        Q,
        K,
        "batch ... queries d_k, batch ... keys d_k -> batch ... queries keys",
    )
    
    pre_softmax_qk = QK / (d_k**0.5)

    # apply mask if provided
    if mask is not None:
        print(pre_softmax_qk.shape, mask.shape)
        # adding a very large negative number to masked positions before softmax
        very_large_negative = -1e9
        pre_softmax_qk = pre_softmax_qk + very_large_negative * (~mask)

    softmax_qk = softmax(pre_softmax_qk, dim=-1)

    attn = einsum(
        softmax_qk,
        V,
        "batch ... queries keys, batch ... keys d_v -> batch ... queries d_v",
    )

    return attn


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_projection_weight: Float[torch.Tensor, "hd_k, d_model"],
        k_projection_weight: Float[torch.Tensor, "hd_k, d_model"],
        v_projection_weight: Float[torch.Tensor, "hd_v, d_model"],
        o_projection_weight: Float[torch.Tensor, "d_model, hd_v"],
        max_seq_len: int | None = None,
        theta: float | None = None,
        token_positions: Int[torch.Tensor, "1 seq"] | None = None,
        use_rope: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model

        self.q_projection_weight = torch.nn.Parameter(
            q_projection_weight.to(device=device, dtype=dtype)
        )
        self.k_projection_weight = torch.nn.Parameter(
            k_projection_weight.to(device=device, dtype=dtype)
        )
        self.v_projection_weight = torch.nn.Parameter(
            v_projection_weight.to(device=device, dtype=dtype)
        )
        self.o_projection_weight = torch.nn.Parameter(
            o_projection_weight.to(device=device, dtype=dtype)
        )

        self.max_seq_len = max_seq_len
        self.theta = theta
        self.token_positions = token_positions
        self.use_rope = use_rope

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
        mask: Bool[torch.Tensor, "seq seq"] | None = None,
    ) -> Float[torch.Tensor, "batch seq d_model"]:

        Q = einsum(
            x,
            self.q_projection_weight,
            "batch seq d_model, hd_k d_model -> batch seq hd_k",
        )

        K = einsum(
            x,
            self.k_projection_weight,
            "batch seq d_model, hd_k d_model -> batch seq hd_k",
        )

        V = einsum(
            x,
            self.v_projection_weight,
            "batch seq d_model, hd_v d_model -> batch seq hd_v",
        )

        # split heads and perform attention
        Q_h = rearrange(
            Q, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads
        )
        K_h = rearrange(
            K, "batch seq (head d_k) -> batch head seq d_k", head=self.num_heads
        )
        V_h = rearrange(
            V, "batch seq (head d_v) -> batch head seq d_v", head=self.num_heads
        )

        if self.use_rope:
            assert (
                self.max_seq_len is not None
                and self.theta is not None
                and self.token_positions is not None
            ), "max_seq_len, theta, token_positions must be provided when use_rope is True"
            rope = RotaryPositionEmbedding(
                theta=self.theta,
                d_k=Q_h.shape[-1],
                max_seq_len=self.max_seq_len,
                device=Q_h.device,
            )
            Q_h = rope(Q_h, self.token_positions)
            K_h = rope(K_h, self.token_positions)

        if mask is None:
            causal_masks = ~torch.triu(
                torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool),
                diagonal=1,
            )
            mask = causal_masks

        # expand mask for batch -> [batch, seq, seq]
        mask_h = rearrange(mask, "q k -> 1 q k").expand(self.num_heads, -1, -1)

        attn = scaled_dot_product_attention(Q_h, K_h, V_h, mask=mask_h)
        attn = rearrange(attn, "batch head seq d_v -> batch seq (head d_v)")

        output = einsum(
            self.o_projection_weight,
            attn,
            "d_model hd_v, batch seq hd_v -> batch seq d_model",
        )

        return output

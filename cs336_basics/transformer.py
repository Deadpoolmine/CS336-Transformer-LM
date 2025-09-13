import torch
from einops import rearrange, einsum

from jaxtyping import Bool, Float, Int

from cs336_basics.attn import MultiheadSelfAttention
from cs336_basics.attn import softmax
from cs336_basics.norm import RMSNorm
from cs336_basics.ffn import FFN
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_d_ff = ffn_d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights
        """
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = RMSNorm(self.d_model)

        # first half of the transformer block
        attn = MultiheadSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            q_projection_weight=self.weights["attn.q_proj.weight"],
            k_projection_weight=self.weights["attn.k_proj.weight"],
            v_projection_weight=self.weights["attn.v_proj.weight"],
            o_projection_weight=self.weights["attn.output_proj.weight"],
            max_seq_len=self.max_seq_len,
            theta=self.theta,
            token_positions=torch.arange(x.shape[-2]),
            use_rope=True,
        )

        norm.load_state_dict({"learned_scale": self.weights["ln1.weight"]})
        attn_x = attn(norm(x))
        x = x + attn_x  # residual connection

        # FFN of the transformer block
        ffn = FFN(
            d_model=self.d_model,
            d_ff=self.ffn_d_ff,
            w1_weight=self.weights["ffn.w1.weight"],
            w2_weight=self.weights["ffn.w2.weight"],
            w3_weight=self.weights["ffn.w3.weight"],
        )

        norm.load_state_dict({"learned_scale": self.weights["ln2.weight"]})
        ffn_x = ffn(norm(x))
        final = x + ffn_x  # residual connection

        return final


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, torch.Tensor],
        vocab_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_d_ff = ffn_d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.weights = weights

        """
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        """

        embed = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        embed.load_state_dict(
            {"embedding_matrix": self.weights["token_embeddings.weight"]}
        )

        layers = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_d_ff=self.ffn_d_ff,
                max_seq_len=self.max_seq_len,
                theta=self.theta,
                weights={
                    k.removeprefix(f"layers.{i}."): v
                    for k, v in self.weights.items()
                    if k.startswith(f"layers.{i}.")
                },
            )
            for i in range(self.num_layers)
        ]

        norm_final = RMSNorm(d_model)
        norm_final.load_state_dict({"learned_scale": self.weights["ln_final.weight"]})

        output_embed = Linear(
            in_features=d_model,
            out_features=vocab_size,
        )
        output_embed.load_state_dict({"weight": self.weights["lm_head.weight"]})

        self.layers = torch.nn.ModuleList(
            [
                embed,
                *layers,
                norm_final,
                output_embed,
            ]
        )
        
        print(num_layers)
        print(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        
        # ??? softmax at the end ???
        return x
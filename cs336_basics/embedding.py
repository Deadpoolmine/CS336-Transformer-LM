import torch
from einops import rearrange, einsum
import math


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        sigma = 1
        torch.nn.init.trunc_normal_(
            self.embedding_matrix, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        Y = self.embedding_matrix[token_ids]
        return Y


class RotaryPositionEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        rotate_blocks = torch.zeros((max_seq_len, d_k, d_k), device=device)

        for i in range(max_seq_len):
            for j in range(d_k // 2):
                angle = i / (theta ** (2 * j / d_k))
                cos_angle = math.cos(angle)
                sin_angle = math.sin(angle)
                rotate_blocks[i, 2 * j, 2 * j] = cos_angle
                rotate_blocks[i, 2 * j, 2 * j + 1] = -sin_angle
                rotate_blocks[i, 2 * j + 1, 2 * j] = sin_angle
                rotate_blocks[i, 2 * j + 1, 2 * j + 1] = cos_angle

        self.register_buffer("rotate_blocks", rotate_blocks, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotate_blocks_buffer = self.get_buffer("rotate_blocks")
        rotate_blocks = rotate_blocks_buffer[token_positions]

        # make sure product in a row-wise fashion
        results = einsum(rotate_blocks, x, "... seq n d_k, ... seq d_k-> ... seq n")

        return results

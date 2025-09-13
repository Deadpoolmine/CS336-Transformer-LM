import torch
from einops import rearrange, einsum


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.learned_scale = torch.nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # perform RMSNorm
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms

        results = einsum(x, self.learned_scale, "... d_model, d_model -> ... d_model")

        return results.to(in_dtype)

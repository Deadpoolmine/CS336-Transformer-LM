import torch
from einops import rearrange, einsum


class FFN(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight,
        w2_weight,
        w3_weight,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(w1_weight.to(device=device, dtype=dtype))
        self.w2_weight = torch.nn.Parameter(w2_weight.to(device=device, dtype=dtype))
        self.w3_weight = torch.nn.Parameter(w3_weight.to(device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # SWIGLU activation
        W1x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        silu = W1x * torch.sigmoid(W1x)
        W3x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        # element-wise multiplication
        hidden = silu * W3x
        # final linear layer
        ffn = einsum(hidden, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")

        return ffn

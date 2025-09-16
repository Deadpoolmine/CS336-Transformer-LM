import torch
from einops import rearrange, einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        sigma = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return Y


if __name__ == "__main__":
    linear = Linear(3, 4)
    x = torch.randn(2, 3)
    y = linear(x)
    print(y)

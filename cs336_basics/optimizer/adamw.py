from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    norm = 0.0
    for p in parameters:
        if p.grad is not None:
            _norm = p.grad.data.norm(2)
            norm += _norm ** 2
    norm = norm**0.5
    
    if norm > max_norm:
        clip_coef = max_norm / (norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return a_max * t / T_w
    elif t >= T_w and t <= T_c:
        return a_min + 0.5 * (a_max - a_min) * (
            1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))
        )
    else:
        return a_min


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "eps": eps,
            "betas": betas,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate
            eps = group["eps"]  # Get the epsilon value
            beta1, beta2 = group["betas"]  # Get the beta values
            weight_decay = group["weight_decay"]  # Get the weight decay value
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.

                t = state.get(
                    "t", 1
                )  # Get iteration number from the state, or initial value.

                m_momentum = state.get(
                    "m_momentum", torch.zeros_like(p.data)
                )  # Initialize 1st moment vector.

                v_momentum = state.get(
                    "v_momentum", torch.zeros_like(p.data)
                )  # Initialize 2nd moment vector.

                g = p.grad.data  # Get the gradient of loss with respect to p.
                m_momentum = beta1 * m_momentum + (1 - beta1) * g
                v_momentum = beta2 * v_momentum + (1 - beta2) * (g * g)

                new_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= new_lr * m_momentum / (torch.sqrt(v_momentum) + eps)
                p.data -= lr * weight_decay * p.data  # Decoupled weight decay
                
                state["t"] = t + 1  # Increment iteration number.
                state["m_momentum"] = m_momentum
                state["v_momentum"] = v_momentum

        return loss

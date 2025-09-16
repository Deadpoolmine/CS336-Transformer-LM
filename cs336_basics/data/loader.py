import torch
import numpy as np
import os
import typing

sample_count = 0


def data_loading(
    x: np.ndarray, batch_size: int, context_length: int, device: torch.device
):
    x_batch = []
    y_batch = []

    for i in range(batch_size):
        start = np.random.randint(0, len(x) - context_length)

        x_batch.append(x[start : start + context_length])
        y_batch.append(x[start + 1 : start + context_length + 1])

    return (
        torch.tensor(np.array(x_batch), device=device),
        torch.tensor(np.array(y_batch), device=device),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

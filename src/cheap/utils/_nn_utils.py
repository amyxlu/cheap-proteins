from typing import Union
import torch
import numpy as np

ArrayLike = Union[np.ndarray, torch.Tensor]


def npy(x: ArrayLike):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


def to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x


def count_parameters(model, require_grad_only=True):
    if require_grad_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def outputs_to_avg_metric(outputs):
    avg_metrics = {}
    metrics_to_log = [
        "plddt",
        "ptm",
        "aligned_confidence_probs",
        "predicted_aligned_error",
    ]

    for metric in metrics_to_log:
        value = npy(outputs[metric])

        if value.ndim == 1:
            median = value
        elif value.ndim == 2:
            median = np.median(value, axis=1)
        else:
            assert value.ndim > 2
            median = np.median(value, axis=tuple(range(1, value.ndim)))

        avg_metrics[metric] = median

    return avg_metrics

import typing as T
import torch
import torch.nn.functional as F
import numpy as np
import einops
import torch


def make_mask(broadcast_shape, mask):
    while len(mask.shape) < len(broadcast_shape):
        mask = mask[..., None]
    return mask.expand(broadcast_shape)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask=None, reduce="mean"):
    """Computes the mean squared error loss.
    assumes that the axis order is (B, L, ...)
    """
    if mask is None:
        return torch.mean((pred - target) ** 2)
    else:
        mask = make_mask(pred.shape, mask)
        if reduce == "mean":
            return ((((pred - target) ** 2) * mask).sum()) / mask.sum()
        elif reduce == "batch":
            dims = tuple(range(1, len(pred.shape)))
            return ((((pred - target) ** 2) * mask).sum(dim=dims)) / mask.sum(dim=dims)
        else:
            raise ValueError(
                f"Unknown reduce type: {reduce}. Expected: 'mean' or 'batch'."
            )


def masked_huber_loss(
    pred: torch.Tensor, target: torch.Tensor, mask=None, reduce="mean"
):
    """Computes the huber loss; assumes that the axis order is (B, L, ...)"""
    if mask is None:
        return F.huber_loss(pred, target)
    else:
        mask = make_mask(pred.shape, mask)
        pred = pred * mask
        target = target * mask
        return F.huber_loss(pred, target, reduction=reduce)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask=None, reduce="mean"):
    """Computes the L1 loss; assumes that the axis order is (B, L, ...)"""
    if mask is None:
        return F.l1_loss(pred, target)
    else:
        mask = make_mask(pred.shape, mask)
        pred = pred * mask
        target = target * mask
        return F.l1_loss(pred, target, reduction=reduce)


def masked_token_cross_entropy_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    mask: T.Optional[torch.Tensor] = None,
    ignore_index: T.Optional[int] = -100,
):
    # pred_logits: (B, L, C) logits.
    # target: (B, L) indices.
    # mask: (B, L) int or bool.
    pred_logits = einops.rearrange(pred_logits, "b l c -> (b l) c")
    targets = einops.rearrange(targets, "b l -> (b l)")

    # The vocab uses 0, which overlaps with the padding idx used by the
    # ESMFold collator, so we use the mask to remove padding positions from
    # array entirely, and then ignore the UNK_IDX when computing the loss.
    if not mask is None:
        mask = einops.rearrange(mask, "b l -> (b l)").to(torch.bool)
        pred_logits = pred_logits[mask, :]
        targets = targets[mask]
    return F.cross_entropy(pred_logits, targets, ignore_index=ignore_index)


def masked_token_accuracy(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    mask: T.Optional[torch.Tensor] = None,
):
    # pred_logits: (B, L, C) logits.
    # target: (B, L) indices.
    # mask: (B, L) int or bool.
    pred_logits = einops.rearrange(pred_logits, "b l c -> (b l) c")
    targets = einops.rearrange(target, "b l -> (b l)")
    if not mask is None:
        mask = einops.rearrange(mask, "b l -> (b l)").to(torch.bool)
        pred_logits = pred_logits[mask, :]
        targets = targets[mask]

    pred = pred_logits.argmax(-1)
    assert pred.shape == targets.shape
    return (pred == targets).sum() / len(pred)

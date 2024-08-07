from typing import List, Tuple

import torch
import random
import einops


def mask_from_seq_lens(x: torch.Tensor, seqlen: torch.Tensor):
    mask = torch.arange(x.shape[1], device=x.device)
    mask = einops.repeat(mask[None, :], "1 L -> N L", N=x.shape[0]) < seqlen[:, None]
    return mask.long()


def get_random_sequence_crop(s, length):
    if len(s) > length:
        start = random.randint(0, len(s) - length)
        return s[start : start + length]
    else:
        return s


def get_random_sequence_crop_batch(sequence_batch, max_len, min_len=None):
    if not min_len is None:
        sequence_batch = list(filter(lambda s: len(s) >= min_len, sequence_batch))
    return [get_random_sequence_crop(seq, max_len) for seq in sequence_batch]


def trim_or_pad_length_first(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (L, ...) to a given length."""
    L = tensor.shape[0]
    if L >= pad_to:
        # trim, assuming first dimension is the dim to trim
        tensor = tensor[:pad_to]
    elif L < pad_to:
        padding = torch.full(
            size=(pad_to - tensor.shape[0], *tensor.shape[1:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=0)
    return tensor


def trim_or_pad_batch_first(tensor: torch.Tensor, pad_to: int, pad_idx: int = 0):
    """Trim or pad a tensor with shape (B, L, ...) to a given length."""
    N, L = tensor.shape[0], tensor.shape[1]
    if L >= pad_to:
        tensor = tensor[:, :pad_to, ...]
    elif L < pad_to:
        padding = torch.full(
            size=(N, pad_to - L, *tensor.shape[2:]),
            fill_value=pad_idx,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.concat((tensor, padding), dim=1)
    return tensor

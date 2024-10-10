import typing as T

import numpy as np

from ._nn_utils import npy
from ..typed import ArrayLike


def calc_sequence_recovery(
    pred_seq: ArrayLike, orig_seq: ArrayLike, mask: T.Optional[ArrayLike] = None
):
    if isinstance(pred_seq[0], str):
        assert isinstance(orig_seq[0], str)
        pred_seq = np.array([ord(x) for x in pred_seq])
        orig_seq = np.array([ord(x) for x in orig_seq])

    if not mask is None:
        pred_seq, orig_seq = pred_seq[mask], orig_seq[mask]
        
    assert len(pred_seq) == len(orig_seq)
    return np.sum(npy(pred_seq) == npy(orig_seq)) / len(pred_seq)
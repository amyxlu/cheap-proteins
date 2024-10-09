from pathlib import Path
import os
import typing as T

import numpy as np
import torch
from torch.hub import download_url_to_file

from ._nn_utils import npy
from ..constants import TENSOR_STATS_DIR, HF_HUB_PREFIX
from ..typed import PathLike


ArrayLike = T.Union[np.ndarray, T.List[float], torch.Tensor]


GLOBAL_SEQEMB_STATS = {
    "uniref": {
        "max": 3038.4783,
        "min": -920.1115,
        "mean": 1.2394488,
        "std": 70.907074,
    },
    "cath": {
        "max": 2853.481,
        "min": -878.217,
        "mean": 1.289,
        "std": 71.788,
    },
}


def _ensure_exists(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)


def _download_normalization_tensors(channel_stat: str, cache_dir: PathLike, dataset: str = "cath", lm_embedder_type="esmfold"):
    target_cache_dir = Path(cache_dir) / f"{dataset}/{lm_embedder_type}/subset_5000_nov28"
    _ensure_exists(target_cache_dir)

    url = f"{HF_HUB_PREFIX}/statistics/cath/esmfold/subset_5000_nov28/channel_{channel_stat}.pkl.npy"
    download_url_to_file(url, target_cache_dir / f"channel_{channel_stat}.pkl.npy")


def _get_npy_path(cache_dir, dataset="cath", lm_embedder_type="esmfold"):
    """Constructs path to the npy file, downloading it if it doesn't exist."""
    assert dataset in ["uniref", "cath"]

    paths = {
        "max": cache_dir
        / dataset
        / lm_embedder_type
        / "subset_5000_nov28"
        / "channel_max.pkl.npy",
        "min": cache_dir
        / dataset
        / lm_embedder_type
        / "subset_5000_nov28"
        / "channel_min.pkl.npy",
        "mean": cache_dir
        / dataset
        / lm_embedder_type
        / "subset_5000_nov28"
        / "channel_mean.pkl.npy",
        "std": cache_dir
        / dataset
        / lm_embedder_type
        / "subset_5000_nov28"
        / "channel_std.pkl.npy",
    }

    for channel_stat, path in paths.items():
        if not path.exists():
            _download_normalization_tensors(channel_stat, cache_dir, dataset, lm_embedder_type)

    return paths


def _array_conversion(
    x: T.Union[float, ArrayLike],
    minv: T.Union[float, ArrayLike],
    maxv: T.Union[float, ArrayLike],
) -> ArrayLike:
    assert type(minv) == type(maxv)

    if isinstance(minv, float):
        return x, minv, maxv

    elif isinstance(x, np.ndarray):
        minv = npy(minv)
        maxv = npy(maxv)
        return x, minv, maxv

    elif isinstance(x, torch.Tensor):
        if isinstance(minv, torch.Tensor):
            minv = minv.to(x.device)
            maxv = maxv.to(x.device)
            return x, minv, maxv
        elif isinstance(minv, np.ndarray):
            # Usually this is the case during training
            minv = torch.from_numpy(minv).to(x.device)
            maxv = torch.from_numpy(maxv).to(x.device)
            return x, minv, maxv

    else:
        raise TypeError("Invalid input type.")


def _minmax_scaling(
    x: ArrayLike,
    data_minv: T.Union[float, ArrayLike],
    data_maxv: T.Union[float, ArrayLike],
    scaled_minv: float = -1.0,
    scaled_maxv: float = 1.0,
) -> ArrayLike:
    """
    Scales all values to between a max and min value, either globally or channel-wise.
    If global, data_minv and data_maxv should be floats. Otherwise,
    they should be ArrayLike denoting the max and min for each channel with shape (1024,)

    The default scaling range is between -1 and 1, following DDPM.
    Follows the sklearn API:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    x, data_minv, data_maxv = _array_conversion(x, data_minv, data_maxv)
    X_std = (x - data_minv) / (data_maxv - data_minv)
    X_scaled = X_std * (scaled_maxv - scaled_minv) + scaled_minv
    return X_scaled


def _undo_minmax_scaling(
    x_scaled: ArrayLike,
    data_minv: T.Union[float, ArrayLike],
    data_maxv: T.Union[float, ArrayLike],
    scaled_minv: float = -1.0,
    scaled_maxv: float = 1.0,
):
    x_scaled, data_minv, data_maxv = _array_conversion(x_scaled, data_minv, data_maxv)
    x_std = (x_scaled - scaled_minv) / (scaled_maxv - scaled_minv)
    x = x_std * (data_maxv - data_minv) + data_minv
    return x


def _standardize(
    x: ArrayLike, meanv: T.Union[float, ArrayLike], stdv: T.Union[float, ArrayLike]
) -> ArrayLike:
    """
    Standardize to center on zero mean with unit std, either globally or channel-wise.
    If global, data_minv and data_maxv should be floats. Otherwise,
    they should be ArrayLike denoting the max and min for each channel with shape (1024,)
    """
    x, meanv, stdv = _array_conversion(x, meanv, stdv)
    return (x - meanv) / stdv


def _undo_standardize(
    x_scaled: ArrayLike,
    meanv: T.Union[float, ArrayLike],
    stdv: T.Union[float, ArrayLike],
):
    x_scaled, meanv, stdv = _array_conversion(x_scaled, meanv, stdv)
    return x_scaled * stdv + meanv


def _scaled_l2_norm(x, scale_factor=1.0):
    """
    Scale to L2 unit norm along channel for each sample, following the
    treatment of CLIP embeddings in DALLE-2.

    Optionally scale up by sqrt(embed_dim), following suggestion from
    https://github.com/lucidrains/DALLE2-pytorch/issues/60
    """
    x = x / x.norm(dim=-1, p="fro", keepdim=True)
    x *= scale_factor
    return x


def _check_valid_mode(mode: str):
    return mode in [
        "global_minmaxnorm",
        "global_standardize",
        "channel_minmaxnorm",
        "channel_standardize",
        "identity",
    ]


def _check_valid_origin_dataset(origin_dataset: str):
    return origin_dataset in ["uniref", "cath"]


def _clamp(tensor: ArrayLike, min_values: ArrayLike, max_values: ArrayLike):
    """
    Clamp values to min/max values defined by an array.
    """
    tensor, min_values, max_values = _array_conversion(tensor, min_values, max_values)
    return torch.where(
        tensor < min_values,
        min_values,
        torch.where(tensor > max_values, max_values, tensor),
    )


def load_channelwise_stats(cache_dir, origin_dataset, lm_embedder_type="esmfold"):
    npy_paths = _get_npy_path(cache_dir, origin_dataset, lm_embedder_type)
    return {
        "max": np.load(npy_paths["max"]),
        "min": np.load(npy_paths["min"]),
        "mean": np.load(npy_paths["mean"]),
        "std": np.load(npy_paths["std"]),
    }


class LatentScaler:
    def __init__(
        self,
        mode: T.Optional[str] = "channel_minmaxnorm",
        origin_dataset: str = "cath",
        lm_embedder_type: str = "esmfold",
    ):
        self.mode = mode
        assert _check_valid_mode(mode), f"Invalid mode {mode}."
        if (mode is None) or (mode == "identity"):
            pass
        else:
            assert _check_valid_origin_dataset(origin_dataset)
            self.mode = mode
            self.origin_dataset = origin_dataset
            self.lm_embedder_type = lm_embedder_type

            if "channel_" in mode:
                stat_dict = load_channelwise_stats(
                    TENSOR_STATS_DIR, origin_dataset, lm_embedder_type
                )
            else:
                stat_dict = GLOBAL_SEQEMB_STATS[origin_dataset]

            self.maxv, self.minv, self.meanv, self.stdv = (
                stat_dict["max"],
                stat_dict["min"],
                stat_dict["mean"],
                stat_dict["std"],
            )

    def scale(self, x: ArrayLike):
        if (self.mode is None) or (self.mode == "identity"):
            return x
        else:
            with torch.no_grad():
                if self.mode == "global_minmaxnorm":
                    x_scaled = _minmax_scaling(x, self.minv, self.maxv)
                elif self.mode == "global_standardize":
                    x_scaled = _standardize(x, self.meanv, self.stdv)
                elif self.mode == "channel_minmaxnorm":
                    x_scaled = _minmax_scaling(x, self.minv, self.maxv)
                elif self.mode == "channel_standardize":
                    x_scaled = _standardize(x, self.meanv, self.stdv)
                else:
                    raise NotImplementedError
            return x_scaled

    def unscale(self, x_scaled: ArrayLike):
        if (self.mode is None) or (self.mode == "identity"):
            return x_scaled
        else:
            with torch.no_grad():
                if self.mode == "global_minmaxnorm":
                    x_scaled = _undo_minmax_scaling(x_scaled, self.minv, self.maxv)
                elif self.mode == "global_standardize":
                    x_scaled = _undo_standardize(x_scaled, self.meanv, self.stdv)
                elif self.mode == "channel_minmaxnorm":
                    x_scaled = _undo_minmax_scaling(x_scaled, self.minv, self.maxv)
                elif self.mode == "channel_standardize":
                    x_scaled = _undo_standardize(x_scaled, self.meanv, self.stdv)
                else:
                    raise NotImplementedError
            return x_scaled


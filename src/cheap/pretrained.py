import os
from pathlib import Path

import torch
from torch.hub import load_state_dict_from_url

from .esmfold import esmfold_v1_embed_only
from .model import HourglassProteinCompressionTransformer
from .pipeline import Pipeline
from .constants import CATH_COMPRESS_LEVEL_TO_ID, CHECKPOINT_DIR_PATH, HF_HUB_PREFIX 
from .typed import PathLike


def url_to_state_dict(url, model_dir):
    """If not already cached, this will download the weights from the given URL and return the state dict."""
    return load_state_dict_from_url(url, model_dir=model_dir, file_name="last.ckpt", progress=True, map_location=torch.device("cpu"))


def load_pretrained_model(
    shorten_factor=1,
    channel_dimension=1024,
    model_dir=CHECKPOINT_DIR_PATH,
    infer_mode=True
):
    if (shorten_factor == 1) and (channel_dimension == 1024):
        # this uses the ESM mechanism for automatically downloading weights if they're not cached
        return esmfold_v1_embed_only()
    else:
        model_id = CATH_COMPRESS_LEVEL_TO_ID[shorten_factor][channel_dimension]
        return load_model_from_id(model_id=model_id, model_dir=model_dir, infer_mode=infer_mode)


def load_model_from_id(
    model_id: str,
    model_dir: PathLike = CHECKPOINT_DIR_PATH,
    infer_mode: bool = True,
):
    url = f"{HF_HUB_PREFIX}/checkpoints/{model_id}/last.ckpt"
    model_dir = Path(model_dir) / model_id

    print(f"Using checkpoint at {str(model_dir)}.")
    ckpt = url_to_state_dict(url, model_dir)

    # initialize model based on saved hyperparameters
    init_hparams = ckpt["hyper_parameters"]
    keys_to_ignore = ["latent_scaler", "seq_emb_fn"]
    for k in keys_to_ignore:
        try:
            init_hparams.pop(k)
        except KeyError:
            pass

    # load state dict
    model = HourglassProteinCompressionTransformer(
        **init_hparams, force_infer=infer_mode
    )
    model.load_state_dict(ckpt["state_dict"])
    if infer_mode:
        model.eval()
        model.requires_grad_(False)

    return model


def get_pipeline(
    model: HourglassProteinCompressionTransformer,
    device: str = "cuda",
):
    esmfold_embed_only = esmfold_v1_embed_only()
    return Pipeline(hourglass_model=model, esmfold_embed_only_module=esmfold_embed_only, device=device)


def CHEAP_shorten_1_dim_1024(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=1024,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_512(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=512,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_256(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=256,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_128(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=128,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_64(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=64,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_32(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=32,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_16(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=16,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_8(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=8,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_1_dim_4(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=1,
        channel_dimension=4,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_1024(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=1024,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_512(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=512,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_256(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=256,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_128(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=128,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_64(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=64,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_32(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=32,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_16(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=16,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_8(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=8,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


def CHEAP_shorten_2_dim_4(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_pretrained_model(
        shorten_factor=2,
        channel_dimension=4,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )
    if return_pipeline:
        return get_pipeline(model)
    return model


# A few 'special cases' compression models


def CHEAP_pfam_shorten_2_dim_32(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH, return_pipeline=True):
    model = load_model_from_id("j1v1wv6w", infer_mode=infer_mode, model_dir=model_dir)
    if return_pipeline:
        return get_pipeline(model)
    return model
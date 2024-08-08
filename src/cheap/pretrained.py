import os
from pathlib import Path

import torch

from .esmfold import esmfold_v1_embed_only
from .model import HourglassProteinCompressionTransformer
from .constants import CATH_COMPRESS_LEVEL_TO_ID, CHECKPOINT_DIR_PATH
from .typed import PathLike


def load_pretrained_model(
    shorten_factor=1,
    channel_dimension=1024,
    model_dir=CHECKPOINT_DIR_PATH,
    infer_mode=True
):
    if (shorten_factor == 1) and (channel_dimension == 1024):
        return esmfold_v1_embed_only()
    else:
        model_id = CATH_COMPRESS_LEVEL_TO_ID[shorten_factor][channel_dimension]
        return load_model_from_id(model_id=model_id, model_dir=model_dir, infer_mode=infer_mode)


def load_model_from_id(
    model_id: str,
    model_dir: PathLike = CHECKPOINT_DIR_PATH,
    infer_mode: bool = True,
):
    checkpoint_fpath = model_dir / model_id / "last.ckpt"
    ckpt = torch.load(checkpoint_fpath)

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


def CHEAP_shorten_1_dim_1024(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=1024,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_512(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=512,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_256(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=256,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_128(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=128,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_64(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=64,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_32(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=32,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_16(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=16,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_8(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=8,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_1_dim_4(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=1,
        channel_dimension=4,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_1024(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=1024,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_512(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=512,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_256(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=256,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_128(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=128,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_64(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=64,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_32(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=32,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_16(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=16,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_8(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=8,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )


def CHEAP_shorten_2_dim_4(infer_mode=True, model_dir=CHECKPOINT_DIR_PATH):
    return load_pretrained_model(
        shorten_factor=2,
        channel_dimension=4,
        infer_mode=infer_mode,
        model_dir=model_dir,
    )

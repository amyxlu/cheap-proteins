import typing as T
from pathlib import Path
import os

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch

import time

from plaid.utils import get_model_device
from plaid.transforms import get_random_sequence_crop_batch

"""
Helper Functions
"""


def make_embedder(lm_embedder_type):
    start = time.time()
    print(f"making {lm_embedder_type}...")

    if "esmfold" in lm_embedder_type:
        # from plaid.denoisers.esmfold import ESMFold
        from plaid.esmfold import esmfold_v1

        embedder = esmfold_v1()
        alphabet = None
    else:
        print("loading LM from torch hub")
        embedder, alphabet = torch.hub.load(
            "facebookresearch/esm:main", lm_embedder_type
        )

    embedder = embedder.eval().to("cuda")

    for param in embedder.parameters():
        param.requires_grad = False

    end = time.time()
    print(f"done loading model in {end - start:.2f} seconds.")

    return embedder, alphabet


def embed_batch_esmfold(esmfold, sequences, max_len=512, embed_result_key="s"):
    with torch.no_grad():
        # don't disgard short sequences since we're also saving headers
        sequences = get_random_sequence_crop_batch(
            sequences, max_len=max_len, min_len=0
        )
        seq_lens = [len(seq) for seq in sequences]
        embed_results = esmfold.infer_embedding(sequences, return_intermediates=True)
        feats = embed_results[embed_result_key].detach()
        seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)
    return feats, seq_lens, sequences


def embed_batch_esm(embedder, sequences, batch_converter, repr_layer, max_len=512):
    sequences = get_random_sequence_crop_batch(sequences, max_len=max_len, min_len=0)
    seq_lens = [len(seq) for seq in sequences]
    seq_lens = torch.tensor(seq_lens, device="cpu", dtype=torch.int16)

    batch = [("", seq) for seq in sequences]
    _, _, tokens = batch_converter(batch)
    device = get_model_device(embedder)
    tokens = tokens.to(device)

    with torch.no_grad():
        results = embedder(tokens, repr_layers=[repr_layer], return_contacts=False)
        feats = results["representations"][repr_layer]

    return feats, seq_lens, sequences


"""
Training 
"""


@hydra.main(
    version_base=None, config_path="configs", config_name="train_sequence_decoder"
)
def train(cfg: DictConfig):
    """
    Set up device and data module
    """
    torch.set_float32_matmul_precision("medium")

    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    max_seq_len = cfg.max_seq_len

    # maybe set up the scaler
    try:
        latent_scaler = hydra.utils.instantiate(cfg.latent_scaler)
        print("scaling")
    except:
        latent_scaler = None
        print("not scaling")

    """
    Set up the embedding model
    """

    lm_embedder_type = cfg.lm_embedder_type
    embedder, alphabet = make_embedder(lm_embedder_type)

    # processing for grabbing intermediates from ESMFold
    if "esmfold" in lm_embedder_type:
        batch_converter = None
        repr_layer = None
        if lm_embedder_type == "esmfold":
            embed_result_key = "s"
        elif lm_embedder_type == "esmfold_pre_mlp":
            embed_result_key = "s_post_softmax"
        else:
            raise ValueError(f"lm embedder type {lm_embedder_type} not understood.")

    # processing for ESM LM-only models
    else:
        batch_converter = alphabet.get_batch_converter()
        repr_layer = int(lm_embedder_type.split("_")[1][1:])
        embed_result_key = None

    """
    Make the embedding function
    """

    embedder = embedder.eval().requires_grad_(False)
    if "esmfold" in lm_embedder_type:
        fn = lambda seqs: embed_batch_esmfold(
            embedder, seqs, max_seq_len, embed_result_key
        )[0]
    else:
        fn = lambda seqs: embed_batch_esm(
            embedder, seqs, batch_converter, repr_layer, max_seq_len
        )

    """
    Run training
    """

    model = hydra.utils.instantiate(
        cfg.sequence_decoder,
        training_embed_from_sequence_fn=fn,
        latent_scaler=latent_scaler,
    )

    job_id = os.environ.get("SLURM_JOB_ID")  # is None if not using SLURM
    dirpath = Path(cfg.paths.checkpoint_dir) / "sequence_decoder" / job_id

    if not cfg.dryrun:
        logger = hydra.utils.instantiate(cfg.logger, id=job_id)
        logger.watch(model, log="all", log_graph=False)
    else:
        logger = None

    lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)
    checkpoint_callback = hydra.utils.instantiate(
        cfg.callbacks.checkpoint, dirpath=dirpath
    )

    trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger, callbacks=[lr_monitor, checkpoint_callback]
    )
    if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
        trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

    if not cfg.dryrun:
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()

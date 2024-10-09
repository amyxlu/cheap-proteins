"""
Load sequences from FASTA, embed by ESMFold, and use specified compression model.

Handles different:
* compression models
* header parsing schemes (For CATH, will parse according to the pattern, otherwise saves the header as is)
* sequence lengths
"""

import time
import typing as T
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import numpy as np
import h5py
from evo.dataset import FastaDataset

from plaid.utils import embed_batch_esmfold, LatentScaler, get_model_device, npy
from plaid.esmfold import esmfold_v1, ESMFold
from plaid.transforms import trim_or_pad_batch_first
from plaid.compression.hourglass_vq import HourglassVQLightningModule

import argparse


PathLike = T.Union[Path, str]


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument("--compressor_model_id", type=str)
    parser.add_argument(
        "--compressor_ckpt_dir",
        type=str,
        default="/data/lux70/cheap/checkpoints/",
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="/data/lux70/data/cath/cath-dataset-nonredundant-S40.atom.fa",
        help="Path to the fasta file",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/data/lux70/data/cath/compressed/",
        help="Directory for training output shards",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Training fraction")
    return parser.parse_args()


def make_hourglass(ckpt_dir: PathLike, ckpt_name: str) -> HourglassVQLightningModule:
    ckpt_path = Path(ckpt_dir) / str(ckpt_name) / "last.ckpt"
    print("Loading hourglass from", str(ckpt_path))
    model = HourglassVQLightningModule.load_from_checkpoint(ckpt_path)
    model = model.eval()
    return model


def make_fasta_dataloaders(
    fasta_file: PathLike, batch_size: int, num_workers: int = 8
) -> T.Tuple[DataLoader, DataLoader]:
    # for loading batches into ESMFold and embedding
    ds = FastaDataset(fasta_file, cache_indices=True)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    return train_dataloader, val_dataloader


def make_shard(
    esmfold: ESMFold,
    hourglass_model: HourglassVQLightningModule,
    dataloader: torch.utils.data.DataLoader,
    max_seq_len: int,
    split_header: bool = True,
    pad_to_even_number: bool = False,
) -> T.Tuple[np.ndarray, T.List[str], T.List[str]]:
    """Set up variables depending on base LM embedder type
    TODO: this will be a sharding function in the future for large, large datasets
    """
    """base loop"""
    cur_compressed = []
    cur_sequences = []
    cur_headers = []
    cur_compression_errors = []

    for batch in tqdm(dataloader, desc="Loop through batches"):
        headers, sequences = batch
        if split_header:
            # for CATH:
            headers = [s.split("|")[2].split("/")[0] for s in headers]

        """
        1. make LM embeddings
        """
        feats, mask, sequences = embed_batch_esmfold(
            esmfold, sequences, max_seq_len, embed_result_key="s", return_seq_lens=False
        )

        """
        2. make hourglass compression
        """
        latent_scaler = LatentScaler()
        x_norm = latent_scaler.scale(feats)
        del feats

        device = get_model_device(hourglass_model)
        x_norm, mask = x_norm.to(device), mask.to(device)

        # for now, this is only for the Rocklin dataset, thus hardcard the target length
        if pad_to_even_number:
            x_norm = trim_or_pad_batch_first(x_norm, pad_to=64)
            mask = trim_or_pad_batch_first(mask, pad_to=64)

        # compressed_representation manipulated in the Hourglass compression module forward pass
        # to return the detached and numpy-ified representation based on the quantization mode.
        recons_norm, loss, log_dict, compressed_representation = hourglass_model(
            x_norm, mask.bool(), log_wandb=False
        )

        """
        3. Ensure correct dtype and append to output list
        """
        compressed_representation = npy(compressed_representation)
        if hourglass_model.quantize_scheme in ["vq", "fsq"]:
            assert compressed_representation.max() >= 65_535
            compressed_representation = compressed_representation.astype(np.uint16)
        else:
            compressed_representation = compressed_representation.astype(np.float32)

        cur_compressed.append(compressed_representation)
        cur_headers.extend(headers)
        cur_sequences.extend(sequences)

    cur_compressed = np.concatenate(cur_compressed, axis=0)
    return cur_compressed, cur_sequences, cur_headers


def save_h5_embeddings(
    embs: np.ndarray,
    sequences: T.List[str],
    pdb_id: str,
    shard_number: int,
    outdir: PathLike,
):
    outdir = outdir
    if not outdir.exists():
        outdir.mkdir(parents=True)

    with h5py.File(str(outdir / f"shard{shard_number:04}.h5"), "w") as f:
        assert isinstance(embs, np.ndarray)
        f.create_dataset("embeddings", data=embs)
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("pdb_id", data=pdb_id)
        print(f"saved {embs.shape[0]} sequences to shard {shard_number} at {str(outdir)} as h5 file")
        print("num unique proteins,", len(np.unique(pdb_id)))
        print("num unique sequences,", len(np.unique(sequences)))
    del embs


def run(
    esmfold: ESMFold,
    compression_model: HourglassVQLightningModule,
    dataloader: DataLoader,
    output_dir: Path,
    cfg: argparse.Namespace,
):
    print(cfg)
    """
    Set up: ESMFold vs. other embedder
    """
    dirname = f"hourglass_{cfg.compressor_model_id}"
    outdir = Path(output_dir) / dirname / f"seqlen_{cfg.max_seq_len}"

    if not outdir.exists():
        outdir.mkdir(parents=True)

    """
    Make shards: wrapper fn
    TODO: for larger datasets, actually shard; here it's just all saved in one file
    """

    compressed, sequences, headers = make_shard(
        esmfold,
        compression_model,
        dataloader,
        cfg.max_seq_len,
        split_header="cath-dataset" in cfg.fasta_file,
        pad_to_even_number="rocklin" in str(output_dir),
    )

    save_h5_embeddings(compressed, sequences, headers, 0, outdir)


def main(cfg):
    """
    Setup: models
    """
    print("loading ESMFold and compression models...")
    start = time.time()
    esmfold = esmfold_v1()
    compression_model = make_hourglass(cfg.compressor_ckpt_dir, cfg.compressor_model_id)
    end = time.time()
    print(f"Models created in {end - start:.2f} seconds.")

    device = torch.device("cuda")
    esmfold.to(device)
    compression_model.to(device)

    print(f"making dataloader from {cfg.fasta_file}")
    train_dataloader, val_dataloader = make_fasta_dataloaders(cfg.fasta_file, cfg.batch_size, cfg.num_workers)

    """
    Run compression and save 
    """
    train_output_dir = Path(cfg.base_output_dir) / "train"
    val_output_dir = Path(cfg.base_output_dir) / "val"

    import IPython

    IPython.embed()
    print("compressing val dataset")
    run(esmfold, compression_model, val_dataloader, val_output_dir, cfg)

    print("compressing train dataset")
    run(esmfold, compression_model, train_dataloader, train_output_dir, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
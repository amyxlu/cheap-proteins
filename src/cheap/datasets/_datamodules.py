import typing as T

import torch
from torch.utils.data import DataLoader

from ..typed import PathLike
from .__precomputed import H5Dataset, StructureH5Dataset
from .__fasta import FastaDataset


class H5DataModule:
    def __init__(
        self,
        shard_dir: PathLike,
        embedder: str = "esmfold",
        seq_len: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,
        dtype: str = "fp32",
        shuffle_val_dataset: bool = False,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.embedder = embedder
        self.dtype = dtype
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset
        self.dataset_fn = H5Dataset

        self.setup()

    def setup(self, stage: str = "fit"):
        kwargs = {}
        kwargs["embedder"] = self.embedder

        if stage == "fit":
            self.train_dataset = self.dataset_fn(
                shard_dir=self.shard_dir,
                split="train",
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
            self.val_dataset = self.dataset_fn(
                shard_dir=self.shard_dir,
                split="val",
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
        elif stage == "predict":
            self.test_dataset = self.dataset_fn(
                split=self.shard_dir,
                shard_dir="val",
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                **kwargs,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class StructureH5DataModule:
    """Loads ground-truth structure as well as embedding for structure-based losses."""

    def __init__(
        self,
        shard_dir: PathLike,
        pdb_path_dir: PathLike,
        embedder: str = "esmfold",
        seq_len: int = 128,
        batch_size: int = 32,
        num_workers: int = 0,
        path_to_filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
        shuffle_val_dataset: bool = False,
    ):
        super().__init__()
        self.shard_dir = shard_dir
        self.pdb_path_dir = pdb_path_dir
        self.embedder = embedder
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dtype = "fp32"
        self.path_to_filtered_ids_list = path_to_filtered_ids_list
        self.max_num_samples = max_num_samples
        self.shuffle_val_dataset = shuffle_val_dataset

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = StructureH5Dataset(
                split="train",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
            self.val_dataset = StructureH5Dataset(
                "val",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
        elif stage == "predict":
            self.test_dataset = StructureH5Dataset(
                "val",
                shard_dir=self.shard_dir,
                pdb_path_dir=self.pdb_path_dir,
                embedder=self.embedder,
                max_seq_len=self.seq_len,
                dtype=self.dtype,
                path_to_filtered_ids_list=self.path_to_filtered_ids_list,
                max_num_samples=self.max_num_samples,
            )
        else:
            raise ValueError(f"stage must be one of ['fit', 'predict'], got {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_val_dataset,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class FastaDataModule:
    def __init__(
        self,
        fasta_file: PathLike,
        batch_size: int,
        train_frac: float = 0.8,
        num_workers: int = 0,
        shuffle_val_dataset: bool = False,
        seq_len: int = 512,
    ):
        self.fasta_file = fasta_file
        self.train_frac, self.val_frac = train_frac, 1 - train_frac
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_val_dataset = shuffle_val_dataset
        self.seq_len = seq_len

    def setup(self):
        ds = FastaDataset(self.fasta_file, cache_indices=True)
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            ds, [self.train_frac, self.val_frac], generator=seed
        )

    def train_dataloader(self, sampler=None):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=(sampler is None),
            sampler=sampler,
        )

    def val_dataloader(self, sampler=None):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle_val_dataset,
            sampler=sampler,
        )

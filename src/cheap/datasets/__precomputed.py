from pathlib import Path
import typing as T

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

from ..utils import StructureFeaturizer
from ..typed import PathLike


ACCEPTED_LM_EMBEDDER_TYPES = [
    "esmfold",  # 1024 -- i.e. t36_3B with projection layers, used for final model
    "esmfold_pre_mlp",  # 2560
    "esm2_t48_15B_UR50D",  # 5120
    "esm2_t36_3B_UR50D",  # 2560
    "esm2_t33_650M_UR50D",  # 1280
    "esm2_t30_150M_UR50D",
    "esm2_t12_35M_UR50D",  # 480
    "esm2_t6_8M_UR50D",  # 320
]


class H5Dataset(Dataset):
    """Loads presaved embeddings as a H5 dataset"""

    def __init__(
        self,
        shard_dir: PathLike,
        split: T.Optional[str] = None,
        embedder: str = "esmfold",
        max_seq_len: int = 64,
        dtype: str = "fp32",
        filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
    ):
        super().__init__()
        self.filtered_ids_list = filtered_ids_list
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.shard_dir = Path(shard_dir)
        self.embedder = embedder
        self.max_num_samples = max_num_samples

        self.data = self.load_partition(
            split, embedder, max_num_samples, filtered_ids_list
        )
        pdb_ids = list(self.data.keys())
        self.pdb_ids = list(pdb_ids)

    def drop_protein(self, pid):
        drop = False
        return drop

    def load_partition(
        self,
        split: T.Optional[str] = None,
        embedder: T.Optional[str] = None,
        max_num_samples: T.Optional[int] = None,
        filtered_ids_list: T.Optional[T.List[str]] = None,
    ):
        """
        2024/02/15: path format:
        ${shard_dir}/${split}/${embedder}/${seqlen}/${precision}/shard0000.h5
        """
        # make sure that the specifications are valid
        datadir = self.shard_dir
        if not split is None:
            assert split in ("train", "val")
            datadir = datadir / split

        if not embedder is None:
            assert embedder in ACCEPTED_LM_EMBEDDER_TYPES
            datadir = datadir / embedder

        datadir = datadir / f"seqlen_{self.max_seq_len}" / self.dtype
        outdict = {}

        # load the shard hdf5 file
        with h5py.File(datadir / "shard0000.h5", "r") as f:
            emb = torch.from_numpy(np.array(f["embeddings"]))
            sequence = list(f["sequences"])
            pdb_ids = list(f["pdb_id"])

            # if prespecified a set of pdb ids, only load those
            if not filtered_ids_list is None:
                pdb_ids = set(pdb_ids).intersection(set(filtered_ids_list))
                disjoint = set(filtered_ids_list) - set(pdb_ids)
                print(
                    f"Did not find {len(disjoint)} IDs, including {list(disjoint)[:3]}, etc."
                )
                pdb_ids = list(pdb_ids)

            # possible trim to a subset to enable faster loading
            if not max_num_samples is None:
                pdb_ids = pdb_ids[:max_num_samples]

            # loop through and decode the protein string one by one
            for i in range(len(pdb_ids)):
                pid = pdb_ids[i].decode()
                if not self.drop_protein(pid):
                    outdict[pid] = (emb[i, ...], sequence[i].decode())

        return outdict

    def __len__(self):
        return len(self.pdb_ids)

    def get(self, idx: int) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(self.pdb_ids, list)
        pid = self.pdb_ids[idx]
        return pid, self.data[pid]

    def __getitem__(
        self, idx: int
    ) -> T.Tuple[str, T.Tuple[torch.Tensor, torch.Tensor]]:
        # wrapper for non-structure dataloaders, rearrange output tuple
        pdb_id, (emb, seq) = self.get(idx)
        return emb, seq, pdb_id


class StructureH5Dataset(H5Dataset):
    """Return ground-truth structure features as well, for structure-based losses."""

    def __init__(
        self,
        shard_dir: PathLike,
        pdb_path_dir: PathLike,
        split: T.Optional[str] = None,
        embedder: str = "esmfold",
        max_seq_len: int = 128,
        dtype: str = "fp32",
        path_to_filtered_ids_list: T.Optional[T.List[str]] = None,
        max_num_samples: T.Optional[int] = None,
    ):
        if not path_to_filtered_ids_list is None:
            with open(path_to_filtered_ids_list, "r") as f:
                filtered_ids_list = f.read().splitlines()
        else:
            filtered_ids_list = None

        super().__init__(
            split=split,
            shard_dir=shard_dir,
            embedder=embedder,
            max_seq_len=max_seq_len,
            dtype=dtype,
            filtered_ids_list=filtered_ids_list,
            max_num_samples=max_num_samples,
        )

        self.structure_featurizer = StructureFeaturizer()
        self.pdb_path_dir = Path(pdb_path_dir)
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx: int):
        pdb_id, (emb, seq) = self.get(idx)
        pdb_path = self.pdb_path_dir / pdb_id
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
        # try:
        #     structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        #     return emb, seq, structure_features
        # except KeyError as e:
        #     with open("bad_ids.txt", "a") as f:
        #         print(pdb_id, e)
        #         f.write(f"{pdb_id}\n")
        #     pass
        structure_features = self.structure_featurizer(pdb_str, self.max_seq_len)
        return emb, seq, structure_features

from typing import (
    Any,
    TypeVar,
    Callable,
    Dict,
    Union,
)
import threading
from pathlib import Path
from operator import methodcaller
import subprocess

import torch
from torch.utils.data import DataLoader
import numpy as np


T = TypeVar("T")
PathLike = Union[str, Path]


"""
Adapted from
https://github.com/rmrao/evo/blob/main/evo/dataset.py
"""


class ThreadsafeFile:
    def __init__(
        self,
        filepath: PathLike,
        open_func: Callable[[PathLike], T],
        close_func: Callable[[T], None] = methodcaller("close"),
    ):
        self._threadlocal = threading.local()
        self._filepath = filepath
        self._open_func = open_func
        self._close_func = close_func

    def __getattr__(self, name: str):
        return getattr(self.file, name)

    @property
    def file(self) -> T:
        if not hasattr(self._threadlocal, "file"):
            self._threadlocal.file = self._open_func(self._filepath)
        return self._threadlocal.file

    def __getstate__(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_threadlocal"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self._threadlocal = threading.local()

    def __del__(self):
        if hasattr(self._threadlocal, "file"):
            self._close_func(self._threadlocal.file)
            del self._threadlocal.file


class SizedDataset(torch.utils.data.Dataset):
    def __init__(self, sizes: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore
        self._sizes = sizes

    def __len__(self):
        return len(self.sizes)

    @property
    def sizes(self):
        return self._sizes


class FastaDataset(SizedDataset):
    """
    For loading protein sequence datasets in the common FASTA data format

    Modified from github.com/pytorch/fairseq.
    """

    def __init__(self, data_file: PathLike, cache_indices: bool = False):
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"{self.data_file}\n"
                "If using hydra, make sure you are using abolute instead of relative paths."
            )
        self.file = ThreadsafeFile(data_file, open)
        self.cache = Path(f"{data_file}.idx.npy")
        if cache_indices:
            if self.cache.exists():
                self.offsets, sizes = np.load(self.cache)
            else:
                self.offsets, sizes = self._build_index()
                np.save(self.cache, np.stack([self.offsets, sizes]))
        else:
            self.offsets, sizes = self._build_index()

        super().__init__(sizes)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx: int):
        self.file.seek(self.offsets[idx])
        if idx == len(self) - 1:
            data = self.file.read()
        else:
            data = self.file.read(self.offsets[idx + 1] - self.offsets[idx])
        desc, *seq = data.split("\n")
        return desc[1:], "".join(seq)

    def __len__(self):
        return self.offsets.size

    def _build_index(self):
        # Use grep and awk to get 100M/s on local SSD.
        # Should process your enormous 100G fasta in ~10 min single core...
        bytes_offsets = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            "| grep --byte-offset '^>' -o | cut -d: -f1",
            shell=True,
        )
        fasta_lengths = subprocess.check_output(
            f"cat {self.data_file} | tqdm --bytes --total $(wc -c < {self.data_file})"
            '| awk \'/^>/ {print "";next;} { printf("%s",$0);}\' | tail -n+2 | awk '
            "'{print length($1)}'",
            shell=True,
        )
        bytes_np = np.fromstring(bytes_offsets, dtype=np.int64, sep=" ")
        sizes_np = np.fromstring(fasta_lengths, dtype=np.int64, sep=" ")
        return bytes_np, sizes_np

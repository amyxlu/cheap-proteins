from typing import Union, List
from pathlib import Path
import torch
import numpy as np

ArrayLike = Union[np.ndarray, torch.Tensor, List]
PathLike = Union[str, Path]

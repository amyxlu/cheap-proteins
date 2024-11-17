import os
from pathlib import Path
from einops import reduce

import torch
from torch import nn

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R
import numpy as np


import string

# disassembled to maintain consistency with
# https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/data/protein.py#L372
# https://github.com/DeepGraphLearning/torchdrug/blob/master/torchdrug/data/protein.py#L50

residue2id = {
    "GLY": 0,
    "ALA": 1,
    "SER": 2,
    "PRO": 3,
    "VAL": 4,
    "THR": 5,
    "CYS": 6,
    "ILE": 7,
    "LEU": 8,
    "ASN": 9,
    "ASP": 10,
    "GLN": 11,
    "LYS": 12,
    "GLU": 13,
    "MET": 14,
    "HIS": 15,
    "PHE": 16,
    "ARG": 17,
    "TYR": 18,
    "TRP": 19,
}
residue_symbol2id = {
    "G": 0,
    "A": 1,
    "S": 2,
    "P": 3,
    "V": 4,
    "T": 5,
    "C": 6,
    "I": 7,
    "L": 8,
    "N": 9,
    "D": 10,
    "Q": 11,
    "K": 12,
    "E": 13,
    "M": 14,
    "H": 15,
    "F": 16,
    "R": 17,
    "Y": 18,
    "W": 19,
}
atom_name2id = {
    "C": 0,
    "CA": 1,
    "CB": 2,
    "CD": 3,
    "CD1": 4,
    "CD2": 5,
    "CE": 6,
    "CE1": 7,
    "CE2": 8,
    "CE3": 9,
    "CG": 10,
    "CG1": 11,
    "CG2": 12,
    "CH2": 13,
    "CZ": 14,
    "CZ2": 15,
    "CZ3": 16,
    "N": 17,
    "ND1": 18,
    "ND2": 19,
    "NE": 20,
    "NE1": 21,
    "NE2": 22,
    "NH1": 23,
    "NH2": 24,
    "NZ": 25,
    "O": 26,
    "OD1": 27,
    "OD2": 28,
    "OE1": 29,
    "OE2": 30,
    "OG": 31,
    "OG1": 32,
    "OH": 33,
    "OXT": 34,
    "SD": 35,
    "SG": 36,
    "UNK": 37,
}
alphabet2id = {
    c: i for i, c in enumerate(" " + string.ascii_uppercase + string.ascii_lowercase + string.digits)
}
id2residue = {v: k for k, v in residue2id.items()}
id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}
id2atom_name = {v: k for k, v in atom_name2id.items()}
id2alphabet = {v: k for k, v in alphabet2id.items()}


def to_sequence(residue_type, num_residues=None):
    """
    Return a sequence of this protein.

    Returns:
        str
    """
    residue_type = residue_type.tolist()
    sequence = []
    if num_residues is None:
        num_residues = len(residue_type)
    for i in range(num_residues):
        sequence.append(id2residue_symbol[residue_type[i]])
    return "".join(sequence)


def to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.tensor(x)

    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.type(dtype)
    return x


def pad_to_multiple(x, shorten_factor):
    from plaid.transforms import trim_or_pad_batch_first

    s = shorten_factor
    extra = x.shape[1] % s
    if extra != 0:
        needed = s - extra
        x = trim_or_pad_batch_first(x, pad_to=x.shape[1] + needed, pad_idx=0)
    return x


@R.register("models.PLAID")
class PLAID(nn.Module, core.Configurable):
    """
    The protein language model, ProtBert-BFD proposed in
    `ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing`_.

    .. _ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing:
        https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf

    Parameters:
        path (str): path to store ProtBert model weights.
        readout (str, optional): readout function. Available functions are ``pooler``, ``sum`` and ``mean``.
    """

    def __init__(
        self,
        compression_model_id="identity",
        hourglass_weights_dir="/data/lux70/cheap/checkpoints",
        pool="mean",
    ):
        super().__init__()
        assert pool in ["mean", "attention"]
        from plaid.compression.hourglass_vq import HourglassVQLightningModule
        from plaid.utils import LatentScaler
        from plaid.esmfold import esmfold_v1

        ckpt_dir = Path(hourglass_weights_dir)
        ckpt_path = ckpt_dir / compression_model_id / "last.ckpt"

        if compression_model_id == "identity":
            self.hourglass = None
            self.shorten_factor = 1
            self.output_dim = 1024
        else:
            self.hourglass = HourglassVQLightningModule.load_from_checkpoint(ckpt_path)
            self.hourglass.eval().requires_grad_(False)
            self.shorten_factor = self.hourglass.enc.shorten_factor
            self.output_dim = 1024 // self.hourglass.enc.downproj_factor

        self.scaler = LatentScaler()
        self.esmfold = esmfold_v1().eval().requires_grad_(False)
        self.pad_idx = 0

    def forward(self, graph, input, all_loss=None, metric=None):
        residues = graph.residue_type
        size = graph.num_residues
        residues, mask = functional.variadic_to_padded(residues, size, value=self.pad_idx)
        mask = mask.to(self.device)

        with torch.no_grad():
            sequences = [to_sequence(residues[i, ...]) for i in range(len(residues))]
            latent = self.esmfold.infer_embedding(sequences)["s"]
            latent = self.scaler.scale(latent)
            latent = latent.to(self.device)

        if not self.hourglass is None:
            with torch.no_grad():
                residue_feature = self.hourglass(latent, mask, infer_only=True)
                residue_feature = to_tensor(residue_feature).to(self.device)
        else:
            residue_feature = to_tensor(latent.detach()).to(self.device)

        # mean pool with mask
        mask = pad_to_multiple(mask, self.shorten_factor)
        downsampled_mask = reduce(mask, "b (n s) -> b n", "sum", s=self.shorten_factor) > 0
        downsampled_mask = downsampled_mask.unsqueeze(-1)

        if downsampled_mask.shape[1] != residue_feature.shape[1]:
            from plaid.transforms import trim_or_pad_batch_first

            downsampled_mask = trim_or_pad_batch_first(downsampled_mask, residue_feature.shape[1], pad_idx=0)
        graph_feature = (residue_feature * downsampled_mask.long()).sum(dim=1) / downsampled_mask.sum(dim=1)

        if self.shorten_factor == 1:
            # hack -- only uses this for the contact prediction tasks, which needs full sequence
            adjusted_size = size // self.shorten_factor
            residue_feature = functional.padded_to_variadic(residue_feature, adjusted_size)
            starts = adjusted_size.cumsum(0) - adjusted_size
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
            residue_feature = residue_feature[mask]

        return {"graph_feature": graph_feature, "residue_feature": residue_feature}


if __name__ == "__main__":
    compression_model_id = "kyytc8i9"
    model = PLAID(compression_model_id)

    from torchdrug import transforms

    truncate_transform = transforms.TruncateProtein(max_length=200, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    from torchdrug import datasets

    dataset = datasets.BetaLactamase(
        "~/protein-datasets/",
        atom_feature=None,
        bond_feature=None,
        residue_feature="default",
        transform=transform,
    )
    train_set, valid_set, test_set = dataset.split()
    print("The label of first sample: ", dataset[0][dataset.target_fields[0]])
    print(
        "train samples: %d, valid samples: %d, test samples: %d"
        % (len(train_set), len(valid_set), len(test_set))
    )

    from torchdrug import tasks

    task = tasks.PropertyPrediction(
        model,
        task=dataset.tasks,
        criterion="mse",
        metric=("mae", "rmse", "spearmanr"),
        normalization=False,
        num_mlp_layer=2,
    )

    import torch
    from torchdrug import core

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, gpus=[0], batch_size=64)
    solver.train(num_epoch=10)
    solver.evaluate("valid")

from pathlib import Path

import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url

from .esmfold import batch_encode_sequences
from .constants import DECODER_CKPT_PATH, HF_HUB_PREFIX


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        n_classes: int = 21,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 3,
        mlp_dropout_p: float = 0.1,
        add_sigmoid: bool = False,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.batch_encode_sequences = batch_encode_sequences
        self.lr = lr

        if mlp_num_layers == 1:
            layers = [nn.Linear(mlp_hidden_dim, n_classes)]

        elif mlp_num_layers == 2:
            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]
            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]
            layers = first_layer + final_layer

        else:
            assert mlp_num_layers >= 3
            num_hidden_layers = mlp_num_layers - 3

            first_layer = [
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            second_layer = [
                nn.Linear(mlp_hidden_dim // 2, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            hidden_layer = [
                nn.Linear(mlp_hidden_dim // 4, mlp_hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=mlp_dropout_p),
            ]

            final_layer = [
                nn.Linear(mlp_hidden_dim // 4, n_classes),
            ]

            layers = (
                first_layer
                + second_layer
                + hidden_layer * num_hidden_layers
                + final_layer
            )

        if add_sigmoid:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # assumes that x is the raw, un-normalized embedding
        return self.net(x)

    @classmethod
    def from_pretrained(cls, device=None, model_dir=None, eval_mode=True):
        if model_dir is None:
            model_dir = Path(DECODER_CKPT_PATH).parent

        url = f"{HF_HUB_PREFIX}/sequence_decoder/mlp.ckpt"

        # will load from cache if available, and otherwise downloads.
        ckpt = load_state_dict_from_url(url, model_dir=model_dir, file_name="mlp.ckpt", progress=True, map_location=torch.device("cpu"))

        model = cls()

        # original model was trained/checkpointed with pytorch lightning
        model.load_state_dict(ckpt["state_dict"])

        if device is not None:
            model.to(device)

        if eval_mode:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model

import torch.nn as nn
import torch
import lightning as L
import typing as T
from pathlib import Path
import os

from typing import Optional, Callable

from .esmfold import batch_encode_sequences
from .constants import DECODER_CKPT_PATH


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
    def from_pretrained(cls, device=None, ckpt_path=None, eval_mode=True):
        if ckpt_path is None:
            ckpt_path = DECODER_CKPT_PATH

        model = cls()

        # original model was trained/checkpointed with pytorch lightning
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])

        if device is not None:
            model.to(device)

        if eval_mode:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model

import typing as T

from openfold.utils.loss import backbone_loss
import pandas as pd
import torch
import wandb

from . import masked_token_cross_entropy_loss, masked_token_accuracy
from ..esmfold._misc import batch_encode_sequences
from ..proteins import LatentToSequence, LatentToStructure
from ..utils import outputs_to_avg_metric


class SequenceAuxiliaryLoss:
    def __init__(
        self,
        sequence_constructor: LatentToSequence,
        weight: float = 1.0,
        loss_fn: T.Callable = masked_token_cross_entropy_loss,
    ):
        self.sequence_constructor = sequence_constructor
        self.loss_fn = loss_fn
        self.weight = weight

    def __call__(
        self,
        latent,
        aatype,
        mask,
        cur_weight=None,
        return_reconstructed_sequences: bool = False,
    ):
        """If cur weight is specified, it will override self.weight."""
        device = latent.device
        self.sequence_constructor.to(device)
        aatype, mask = aatype.to(device), mask.to(device)

        # grab logits and calculate masked cross entropy (must pass non-default arguments)
        logits, _, recons_strs = self.sequence_constructor.to_sequence(
            latent, mask, return_logits=True, drop_mask_idx=False
        )
        loss = self.loss_fn(logits, aatype, mask)
        acc = masked_token_accuracy(logits, aatype, mask)
        weight = self.weight if cur_weight is None else cur_weight
        logdict = {
            "seq_loss": loss.item(),
            "seq_acc": acc.item(),
        }
        if return_reconstructed_sequences:
            return weight * loss, logdict, recons_strs
        else:
            return (
                weight * loss,
                logdict,
            )


class BackboneAuxiliaryLoss:
    def __init__(self, structure_constructor: LatentToStructure, weight=1.0):
        self.structure_constructor = structure_constructor
        self.weight = weight

    def __call__(
        self,
        latent,
        gt_structures,
        sequences,
        num_recycles=1,
        inner_batch_size=None,
        cur_weight=None,
    ):
        device = latent.device
        self.structure_constructor.to(device)

        # check shapes
        batch_size, seq_len, _ = latent.shape
        assert gt_structures["backbone_rigid_tensor"].shape == torch.Size(
            [batch_size, seq_len, 4, 4]
        )
        assert gt_structures["backbone_rigid_mask"].shape == torch.Size(
            [batch_size, seq_len]
        )

        # todo: maybe also log pdb strs
        # pred_structures = self.trunk.from_seq_feat(true_aa, latent)[0]
        pred_pdb_strs, pred_raw_outputs = self.structure_constructor.to_structure(
            latent,
            sequences,
            num_recycles,
            batch_size=inner_batch_size,
            return_raw_features=True,
        )
        assert pred_raw_outputs["frames"].shape == torch.Size(
            [8, batch_size, seq_len, 7]
        )

        loss = backbone_loss(
            backbone_rigid_tensor=gt_structures["backbone_rigid_tensor"].to(device),
            backbone_rigid_mask=gt_structures["backbone_rigid_mask"].to(device),
            traj=pred_raw_outputs["frames"],
        )

        weight = self.weight if cur_weight is None else cur_weight
        metrics = outputs_to_avg_metric(pred_raw_outputs)
        logdict = {"backbone_loss": loss.item()} | metrics
        return weight * loss, logdict

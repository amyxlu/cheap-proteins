import typing as T

import torch
from torch import nn
import numpy as np

from . import HourglassDecoder, HourglassEncoder, VectorQuantizer, FiniteScalarQuantizer
from ..utils import (
    LatentScaler,
    trim_or_pad_batch_first,
    get_lr_scheduler,
    get_model_device,
)
from ..esmfold._misc import batch_encode_sequences
from ..proteins import LatentToSequence, LatentToStructure
from ..losses import SequenceAuxiliaryLoss, BackboneAuxiliaryLoss, masked_mse_loss


class HourglassProteinCompressionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth=4,  # depth used for both encoder and decoder
        shorten_factor=2,
        downproj_factor=2,
        attn_resampling=True,
        updown_sample_type="naive",
        heads=8,
        dim_head=64,
        causal=False,
        norm_out=False,
        use_quantizer="vq",
        # quantizer
        n_e=512,
        e_dim=64,
        vq_beta=0.25,
        enforce_single_codebook_per_position: bool = False,
        fsq_levels: T.Optional[T.List[int]] = None,
        lr=1e-4,
        lr_adam_betas=(0.9, 0.999),
        lr_sched_type: str = "constant",
        lr_num_warmup_steps: int = 0,
        lr_num_training_steps: int = 10_000_000,
        lr_num_cycles: int = 1,
        # auxiliary losses
        seq_loss_weight: float = 0.0,
        struct_loss_weight: float = 0.0,
        log_sequence_loss=False,
        log_structure_loss=False,
        # in case we need to embed on the fly
        esmfold=None,
        force_infer=False,
    ):
        super().__init__()

        """Make quantizer. Can be either the traditional VQ-VAE, the FSQ, or
        none (i.e. output of encoder goes directly back into the decoder).
        """

        self.latent_scaler = LatentScaler()

        if esmfold is not None:
            self.esmfold = esmfold
            for param in self.esmfold.parameters():
                param.requires_grad = False

        if isinstance(use_quantizer, bool):
            if use_quantizer:
                # for backwards compatitibility
                print("Using quantizer: VQVAE")
                self.quantize_scheme = "vq"
            else:
                # for backwards compatitibility
                print("Using non-quantization mode...")
                self.quantize_scheme = None  # no quantization
        else:
            assert use_quantizer in ["vq", "fsq", "tanh"]
            self.quantize_scheme = use_quantizer
            print(f"Using {use_quantizer} layer at bottleneck...")

        assert self.check_valid_compression_method(self.quantize_scheme)

        # Set up quantizer modules
        self.pre_quant_proj = None
        self.post_quant_proj = None

        if self.quantize_scheme == "vq":
            self.quantizer = VectorQuantizer(n_e, e_dim, vq_beta)

            # if this is enforced, then we'll project down the channel dimension to make sure that the
            # output of the encoder has the same dimension as the embedding codebook.
            # otherwise, the excess channel dimensions will be tiled up lengthwise,
            # which combinatorially increases the size of the codebook. The latter will
            # probably lead to better results, but is not the convention and may lead to
            # an excessively large codebook for purposes such as training an AR model downstream.
            if enforce_single_codebook_per_position and (
                dim / downproj_factor != e_dim
            ):
                self.pre_quant_proj = torch.nn.Linear(dim // downproj_factor, e_dim)
                self.post_quant_proj = torch.nn.Linear(e_dim, dim // downproj_factor)

        elif self.quantize_scheme == "fsq":
            if not len(fsq_levels) == (dim / downproj_factor):
                # similarly, project down to the length of the FSQ vectors.
                # unlike with VQ-VAE, the convention with FSQ *is* to combinatorially incraese the size of codebook
                self.pre_quant_proj = torch.nn.Linear(
                    dim // downproj_factor, len(fsq_levels)
                )
                self.post_quant_proj = torch.nn.Linear(
                    len(fsq_levels), dim // downproj_factor
                )
            self.fsq_levels = fsq_levels
            self.quantizer = FiniteScalarQuantizer(fsq_levels)
        else:
            # self.quantize_scheme in [None, "tanh"]
            self.quantizer = None

        # Set up encoder/decoders
        self.enc = HourglassEncoder(
            dim=dim,
            depth=depth,
            shorten_factor=shorten_factor,
            downproj_factor=downproj_factor,
            attn_resampling=attn_resampling,
            updown_sample_type=updown_sample_type,
            heads=heads,
            dim_head=dim_head,
            causal=causal,
            norm_out=norm_out,
        )
        self.dec = HourglassDecoder(
            dim=dim // downproj_factor,
            depth=depth,
            elongate_factor=shorten_factor,
            upproj_factor=downproj_factor,
            attn_resampling=True,
            updown_sample_type=updown_sample_type,
        )

        # other misc settings
        self.z_q_dim = dim // np.prod(dim)
        self.n_e = n_e

        self.lr = lr
        self.lr_adam_betas = lr_adam_betas
        self.lr_sched_type = lr_sched_type
        self.lr_num_warmup_steps = lr_num_warmup_steps
        self.lr_num_training_steps = lr_num_training_steps
        self.lr_num_cycles = lr_num_cycles

        self.make_sequence_constructor = log_sequence_loss or (seq_loss_weight > 0.0)
        self.make_structure_constructor = log_structure_loss or (
            struct_loss_weight > 0.0
        )
        self.seq_loss_weight = seq_loss_weight
        self.struct_loss_weight = struct_loss_weight

        # auxiliary losses
        if not force_infer:
            if self.make_sequence_constructor:
                self.sequence_constructor = LatentToSequence()
                self.seq_loss_fn = SequenceAuxiliaryLoss(self.sequence_constructor)

            if self.make_structure_constructor:
                self.structure_constructor = LatentToStructure(esmfold=esmfold)
                self.structure_loss_fn = BackboneAuxiliaryLoss(
                    self.structure_constructor
                )

        print(
            f"Finished loading HPCT model with shorten factor {shorten_factor} and {1024 // downproj_factor} channel dimensions."
        )

    def check_valid_compression_method(self, method):
        return method in ["fsq", "vq", "tanh", None]

    def encode(self, x, mask=None, verbose=False, infer_only=True, *args, **kwargs):
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[1])).to(x.device)

        mask = mask.bool()

        # ensure that input length is a multiple of the shorten factor
        s = self.enc.shorten_factor
        extra = x.shape[1] % s
        if extra != 0:
            needed = s - extra
            x = trim_or_pad_batch_first(x, pad_to=x.shape[1] + needed, pad_idx=0)

        # In any case where the mask and token generated from sequence strings don't match latent, make it match
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)

        # encode and possibly downsample
        log_dict = {}
        z_e, downsampled_mask = self.enc(x, mask, verbose)

        # if encoder output dimensions does not match quantization inputs, project down
        if self.pre_quant_proj is not None:
            z_e = self.pre_quant_proj(z_e)

        ##################
        # Quantize
        ##################

        # VQ-VAE

        if self.quantize_scheme == "vq":
            quant_out = self.quantizer(z_e, verbose)
            if not infer_only:
                z_q = quant_out["z_q"]
                vq_loss = quant_out["loss"]
                log_dict["vq_loss"] = quant_out["loss"]
                log_dict["vq_perplexity"] = quant_out["perplexity"]
                compressed_representation = quant_out[
                    "min_encoding_indices"
                ].detach()  # .cpu().numpy()

        # FSQ

        elif self.quantize_scheme == "fsq":
            z_q = self.quantizer.quantize(z_e)
            compressed_representation = self.quantizer.codes_to_indexes(
                z_q
            ).detach()  # .cpu().numpy()

        # Continuous (no quantization) with a tanh bottleneck

        elif self.quantize_scheme == "tanh":
            z_e = z_e.to(torch.promote_types(z_e.dtype, torch.float32))
            z_q = torch.tanh(z_e)
        
        else:
            raise NotImplementedError

        if infer_only:
            compressed_representation = z_q.detach()  # .cpu().numpy()
            downsampled_mask = downsampled_mask.detach()  # .cpu().numpy()
            return compressed_representation, downsampled_mask
        else:
            return z_q, downsampled_mask, log_dict

    def decode(self, z_q, downsampled_mask=None, verbose=False):
        if self.post_quant_proj is not None:
            z_q = self.post_quant_proj(z_q)

        x_recons = self.dec(z_q, downsampled_mask, verbose)
        return x_recons

    def forward(self, x, mask=None, verbose=False, infer_only=True, *args, **kwargs):
        if infer_only:
            return self.encode(x, mask, verbose, infer_only)
        
        else:
            # encode and obtain post-quantize embedding
            z_q, downsampled_mask, log_dict = self.encode(x, mask, verbose, infer_only)

            # decode back to original
            x_recons = self.decode(z_q, downsampled_mask, verbose)

            # calculate losses
            recons_loss = masked_mse_loss(x_recons, x, mask)
            vq_loss = log_dict.get("vq_loss", 0.0)
            loss = vq_loss + recons_loss
            log_dict["recons_loss"] = recons_loss.item()
            log_dict["loss"] = loss.item()
        
            return x_recons, loss, log_dict, z_q, downsampled_mask

    def configure_optimizers(self):
        parameters = list(self.enc.parameters()) + list(self.dec.parameters())
        if not self.quantizer is None:
            parameters += list(self.quantizer.parameters())

        optimizer = torch.optim.AdamW(
            parameters, lr=self.lr, betas=self.lr_adam_betas
        )
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            sched_type=self.lr_sched_type,
            num_warmup_steps=self.lr_num_warmup_steps,
            num_training_steps=self.lr_num_training_steps,
            num_cycles=self.lr_num_cycles,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def run_batch(self, batch, prefix="train"):
        """
        The input batch can be:
        (1) precomputed embeddings along with a dictionary of structures (CATHShardedDataModule)
        (2) precomputed embeddings with a placeholder for the structure dictionary (CATHStructureDataModule)
        (3) raw headers and sequences tuples (FastaDataset)

        to trigger the raw sequence mode, the `seq_emb_fn` should be passed, which should be defined outside
        the train loop, and should of the desired embedding function from ESMFold/etc., already moved to device.
        """

        # hack infer which type of input batch we're using
        if len(batch) == 3:
            x, sequences, gt_structures = batch

        elif len(batch) == 2:
            # using a FastaLoader, sequence only
            assert not self.esmfold is None
            headers, sequences = batch
            x = self.esmfold.infer_embedding(sequences)["s"]

        else:
            raise

        device = get_model_device(self)

        # get masks and ground truth tokens and move to device
        tokens, mask, _, _, _ = batch_encode_sequences(sequences)

        # if shortened and using a Fasta loader, the latent might not be a multiple of shorten factor
        s = self.enc.shorten_factor
        extra = x.shape[1] % s
        if extra != 0:
            needed = s - extra
            x = trim_or_pad_batch_first(x, pad_to=x.shape[1] + needed, pad_idx=0)

        # In any case where the mask and token generated from sequence strings don't match latent, make it match
        if mask.shape[1] != x.shape[1]:
            # pad with False
            mask = trim_or_pad_batch_first(mask, x.shape[1], pad_idx=0)
            tokens = trim_or_pad_batch_first(tokens, x.shape[1], pad_idx=0)

        x = x.to(device)
        mask = mask.to(device)
        tokens = tokens.to(device)

        # scale (maybe) latent values to be per-channel normalized
        x = self.latent_scaler.scale(x)

        # forward pass
        log_dict = {}
        x_recons, loss, log_dict, _, _ = self(x, mask.bool())
        log_dict = log_dict | {f"{prefix}/{k}": v for k, v in log_dict.items()}

        # unscale to decode into sequence and/or structure
        x_recons_unscaled = self.latent_scaler.unscale(x_recons)
        batch_size = x_recons_unscaled.shape[0]

        # sequence loss
        if self.make_sequence_constructor:
            self.sequence_constructor = self.sequence_constructor.to(device)
            with torch.no_grad():
                seq_loss, seq_loss_dict, recons_strs = self.seq_loss_fn(
                    x_recons_unscaled, tokens, mask, return_reconstructed_sequences=True
                )
            seq_loss_dict = {f"{prefix}/{k}": v for k, v in seq_loss_dict.items()}
            log_dict = log_dict | seq_loss_dict
            loss += seq_loss * self.seq_loss_weight

        # structure loss
        if self.make_structure_constructor:
            self.structure_constructor = self.structure_constructor.to(device)
            with torch.no_grad():
                struct_loss, struct_loss_dict = self.structure_loss_fn(
                    x_recons_unscaled, gt_structures, sequences
                )
            struct_loss_dict = {
                f"{prefix}/{k}": v.mean() for k, v in struct_loss_dict.items()
            }
            log_dict = log_dict | struct_loss_dict
            loss += struct_loss * self.struct_loss_weight

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.run_batch(batch, prefix="val")

    def state_dict(self):
        state = super().state_dict()
        state = {k: v for k, v in state.items() if "esmfold" not in k}
        return state

    @classmethod
    def from_pretrained(cls, checkpoint_path):
        model = cls()
        state = torch.load(checkpoint_path)
        model.load_state_dict(state)
        return model

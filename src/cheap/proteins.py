import os
import re
import typing as T
from pathlib import Path
from tqdm import trange

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

import torch
import typing as T
import numpy as np
import re
from openfold.np import residue_constants
from lightning.pytorch.utilities import rank_zero_info

from .utils import npy, to_tensor, get_model_device, trim_or_pad_batch_first
from .decoder import FullyConnectedNetwork
from .esmfold import ESMFOLD_Z_DIM, esmfold_v1, output_to_pdb, batch_encode_sequences
from .typed import PathLike, ArrayLike


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# https://github.com/dauparas/ProteinMPNN/blob/main/protein_mpnn_utils.py#L61
PROTEINMPNN_AACHAR_TO_AAIDX_ARR = list("ARNDCQEGHILKMFPSTWYV-")
PROTEINMPNN_AAIDX_TO_AACHAR = {
    idx: char for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)
}
PROTEINMPNN_AACHAR_TO_AAIDX = {
    char: idx for idx, char in enumerate(PROTEINMPNN_AACHAR_TO_AAIDX_ARR)
}


def stack_tensor_dicts(
    dicts: T.List[T.Dict[str, torch.Tensor]], list_of_igored_keys: T.List[str]
):
    keys = set(k for d in dicts for k in d.keys())
    keys = keys - set(list_of_igored_keys)
    return {key: torch.cat([d[key] for d in dicts if key in d], dim=0) for key in keys}


class DecoderTokenizer:
    def __init__(self, vocab="openfold"):
        if vocab == "openfold":
            self.aachar_to_aaidx = residue_constants.restype_order_with_x
            self.aaidx_to_aachar = {v: k for k, v in self.aachar_to_aaidx.items()}
        elif vocab == "proteinmpnn":
            self.aachar_to_aaidx = PROTEINMPNN_AACHAR_TO_AAIDX
            self.aaidx_to_aachar = PROTEINMPNN_AAIDX_TO_AACHAR
        else:
            raise ValueError(f"Unknown vocab {vocab}")

        self.vocab_size = len(self.aaidx_to_aachar)
        self.unk_idx = self.aachar_to_aaidx.get("X", None)
        self.pad_idx = self.aachar_to_aaidx.get("_", None)
        self.eos_idx = self.aachar_to_aaidx.get(">", None)
        self.bos_idx = self.aachar_to_aaidx.get("<", None)

    def __len__(self):
        return len(self.aaidx_to_aachar)

    def _char2idx(self, char: str) -> int:
        return self.aachar_to_aaidx.get(char, self.unk_idx)

    def str_to_aatype_sequence(
        self, seq: T.Union[T.Iterable, str], as_torch_tensor: bool = True
    ):
        if isinstance(seq, str):
            seq = list(seq)

        aatype = [self._char2idx(aa) for aa in seq]
        if as_torch_tensor:
            return torch.tensor(aatype)
        else:
            return aatype

    def aatype_to_str_sequence(self, aatype: T.List[int], strip_mode: str = "none"):
        assert strip_mode in ["none", "strip_pad", "strip_eos", "strip_after_eos"]
        aastr = "".join([self.aaidx_to_aachar[aa] for aa in npy(aatype)])
        if strip_mode == "none":
            return aastr
        elif strip_mode == "strip_pad":
            aastr = aastr.replace("_", "")
        elif strip_mode == "strip_eos":
            # strip ">" and everything after it
            pattern = r"^(.*?)[>]"
        elif strip_mode == "strip_after_eos":
            # keep ">" but strip everything after it
            pattern = r"^(.*?[>])"
            match = re.search(pattern, aastr)
            if match:
                aastr = match.group(1)
        else:
            raise ValueError(f"Unrecognized strip_mode: {strip_mode}")
        return aastr

    def collate_dense_tensors(
        self, samples: T.List[torch.Tensor], pad_v: int
    ) -> torch.Tensor:
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
            (d_21,       ...,           d_2K),
            ...,
            (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """

        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result

    def batch_encode_sequences(
        self, sequences: T.Sequence[str], pad_v: T.Optional[int] = None
    ) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences using this tokenization scheme, mostly to generate labels during training
        of the autoregressive decoder.

        Args:
            sequences (T.Sequence[str]): List of amino acid sequence strings.
            add_eos_bos (bool): Whether or not to also add the local EOS/BOS token in generating the labels.

        Returns:
            T.Tuple[torch.Tensor, torch.Tensor]: Amino acid indices and mask (0 if padded, 1 otherwise).
        """
        if pad_v is None:
            pad_v = self.pad_idx

        aatype_list = []

        for seq in sequences:
            aatype_seq = self.str_to_aatype_sequence(seq)
            aatype_list.append(aatype_seq)

        aatype = self.collate_dense_tensors(aatype_list, pad_v=pad_v)
        mask = self.collate_dense_tensors(
            [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list],
            pad_v=pad_v,
        )
        return aatype, mask

    def _is_valid_aa(self, aa):
        return aa in CANONICAL_AA

    def remove_invalid_aa(self, string: str):
        return "".join([s for s in string if self._is_valid_aa(s)])


class LatentToSequence:
    def __init__(self, temperature: float = 1.0):
        """On construction, all models are on the CPU."""
        self.temperature = temperature
        self.tokenizer = DecoderTokenizer()
        self.decoder = FullyConnectedNetwork.from_pretrained(device="cpu")
        self.device = torch.device("cpu")

        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False

    def to(self, device):
        """Move onto the device for the usecase before calling to_sequence()."""
        self.decoder = self.decoder.to(device)
        self.device = device
        return self

    def to_sequence(
        self, latent: ArrayLike, mask=None, return_logits=False, drop_mask_idx=True
    ):
        if not mask is None:
            mask = torch.ones_like(latent)
        latent = to_tensor(latent, device=self.device)
        assert (
            latent.device == get_model_device(self.decoder)
        ), "Make sure to call .to(device) to move decoder to the correct device."

        with torch.no_grad():
            output_logits = self.decoder(latent)

        # adjust by temperature
        output_logits /= self.temperature

        # remove UNK token
        if drop_mask_idx:
            _mask = (
                torch.arange(output_logits.shape[-1], device=self.device)
                != self.tokenizer.unk_idx
            )
            drop_mask_logits = torch.index_select(
                input=output_logits,
                dim=-1,
                index=torch.arange(output_logits.shape[-1], device=self.device)[_mask],
            )
            argmax_idx = drop_mask_logits.argmax(-1)
            dist = torch.distributions.OneHotCategorical(logits=drop_mask_logits)
            sequence_probs = F.softmax(drop_mask_logits, dim=-1)
        else:
            # get the argmax index & compare it to the actual sample, to get a sense as to how temperature affects diversity
            argmax_idx = output_logits.argmax(-1)
            dist = torch.distributions.OneHotCategorical(logits=output_logits)
            sequence_probs = F.softmax(output_logits, dim=-1)

        sequence_idx = dist.sample().argmax(-1)
        sequence_probs = torch.gather(
            sequence_probs, dim=-1, index=argmax_idx.unsqueeze(-1)
        ).squeeze(-1)
        stochasticity = (argmax_idx == sequence_idx).sum() / torch.numel(argmax_idx)
        # print(f"percentage similarty to argmax idx: {stochasticity:.3f}")

        sequence_str = [
            self.tokenizer.aatype_to_str_sequence(s)
            for s in sequence_idx.long().cpu().numpy()
        ]

        if return_logits:
            # return the original output logits, e.g. for loss & backprop purposes
            return output_logits, sequence_idx, sequence_str
        else:
            return sequence_probs, sequence_idx, sequence_str


class LatentToStructure:
    def __init__(self, esmfold=None, chunk_size=128, delete_esm_lm=False, use_compile=False):
        if esmfold is None:
            esmfold = esmfold_v1()

        self.esmfold = esmfold
        self.esmfold.set_chunk_size(chunk_size)
        if delete_esm_lm:
            del self.esmfold.esm  # save some GPU space
        assert not self.esmfold.trunk is None

        self.esmfold.eval()
        for param in self.esmfold.parameters():
            param.requires_grad = False
        
        if use_compile:
            self.esmfold = torch.compile(self.esmfold)
            
        self.device = get_model_device(self.esmfold)

    def to(self, device):
        self.esmfold = self.esmfold.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def run_batch(self, s_, aa_, mask_, residx_, num_recycles=1, *args, **kwargs):
        # https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py#L208
        # utils.print_cuda_memory_usage()
        _, L, _ = s_.shape
        z_ = s_.new_zeros(s_.shape[0], L, L, ESMFOLD_Z_DIM).to(self.device)

        def maybe_pad(tensor, length):
            if tensor.shape[1] != length:
                return trim_or_pad_batch_first(tensor, length, pad_idx=0)
            else:
                return tensor

        mask_ = maybe_pad(mask_, L)
        aa_ = maybe_pad(aa_, L)
        residx_ = maybe_pad(residx_, L)

        with torch.no_grad():
            output = self.esmfold.folding_trunk(
                s_s_0=s_,
                s_z_0=z_,
                aa=aa_,
                residx=residx_,
                mask=mask_,
                num_recycles=num_recycles,
            )
        pdb_str = output_to_pdb(output)

        for k, v in output.items():
            try:
                output[k] = v.cpu()
            except:
                pass
        return pdb_str, output

    def to_structure(
        self,
        latent: ArrayLike,
        sequences: T.List[str],
        num_recycles: int = 4,
        batch_size: T.Optional[int] = None,
        mask: T.Optional[ArrayLike] = None,  # used to override padding positions if sequence includes padding.
        return_raw_outputs: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> T.Tuple[T.List[PathLike], T.Union[T.Dict, pd.DataFrame]]:
        """set up devices and tensors"""
        aatype, mask_, residx, _, _ = batch_encode_sequences(sequences)

        if mask is None:
            mask = mask_

        aatype, mask, residx = tuple(
            map(lambda x: x.to(self.device), (aatype, mask, residx))
        )
        latent = to_tensor(latent, device=self.device)
        assert (
            latent.device == self.esmfold.device
        ), "Make sure to call .to(device) to move trunk to the correct device."

        if batch_size is None:
            if verbose:
                print("Generating structure from latents")
            return self.run_batch(latent, aatype, mask, residx, num_recycles)

        else:
            all_output_dicts = []
            all_pdb_strs = []
            for start in trange(
                0, len(latent), batch_size, desc="(Generating structure)"
            ):
                # Process current batch
                s_, aa_, mask_, residx_ = tuple(
                    map(
                        lambda x: x[start : start + batch_size],
                        (latent, aatype, mask, residx),
                    )
                )

                # Collect outputs
                pdb_str, outputs = self.run_batch(s_, aa_, mask_, residx_, num_recycles)
                all_pdb_strs.extend(pdb_str)
                all_output_dicts.append(outputs)

            if return_raw_outputs:
                try:
                    all_output_dicts = stack_tensor_dicts(
                        all_output_dicts,
                        list_of_igored_keys=["max_predicted_aligned_error"],
                    )
                except:
                    rank_zero_info(
                        "Error stacking tensors from batches, returning raw list of outputs instead."
                    )
                return all_pdb_strs, all_output_dicts
            else:
                return all_pdb_strs

"""
CHEAP model pipeline wrapper around ESMFold embedding, normalization module, and hourglass compression.
"""

from typing import Optional, Union, List, Tuple

import torch

from .model import HourglassProteinCompressionTransformer
from .esmfold import ESMFoldEmbed, esmfold_v1_embed_only 
from .utils import LatentScaler
from .typed import DeviceLike


class Pipeline:
    def __init__(
            self,
            hourglass_model: HourglassProteinCompressionTransformer,
            esmfold_embed_only_module: ESMFoldEmbed,
            latent_scaler: LatentScaler = LatentScaler(),
            device: DeviceLike = "cuda",
        ):
        super().__init__()
        self.hourglass_model = hourglass_model.to(device)
        self.esmfold_embed_only_module = esmfold_embed_only_module.to(device)
        self.latent_scaler = latent_scaler
        self.device = device
    
    def to(self, device: DeviceLike):
        self.hourglass_model = self.hourglass_model.to(device)
        self.esmfold_embed_only_module = self.esmfold_embed_only_module.to(device)
        self.device = device
        return self
    
    def decode(self, x_compressed: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Given a compressed representation, uncompress and unnormalize back to the original ESMFold latent."""
        x_uncompressed = self.hourglass_model.decode(x_compressed, mask) 
        return self.latent_scaler.unscale(x_uncompressed)
    
    def __call__(self, sequences: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given the original ESMFold latent, normalize and compress using the loaded checkpoint."""
        res = self.esmfold_embed_only_module.infer_embedding(sequences)
        emb, mask = res['s'], res['mask']
        emb, mask = emb.to(self.device), mask.to(self.device)

        emb = self.latent_scaler.scale(emb)
        compressed_representation, downsampled_mask = self.hourglass_model(emb, mask, infer_only=True)
        return compressed_representation, downsampled_mask

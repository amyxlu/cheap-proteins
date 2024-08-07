import typing as T
from pathlib import Path

import numpy as np
import torch

from ..openfold_utils import (
    make_pdb_features,
    make_all_atom_aatype,
    make_seq_mask,
    make_atom14_masks,
    make_atom14_positions,
    atom37_to_frames,
    get_backbone_frames,
    OFProtein,
    protein_from_pdb_string,
)
from ._transforms import trim_or_pad_length_first

PathLike = T.Union[Path, str]


FEATURES_REQUIRING_PADDING = [
    "aatype",
    "between_segment_residues",
    "residue_index",
    "seq_length",
    "all_atom_positions",
    "all_atom_mask",
    #  'resolution',
    #  'is_distillation',
    "all_atom_aatype",
    "seq_mask",
    "atom14_atom_exists",
    "residx_atom14_to_atom37",
    "residx_atom37_to_atom14",
    "atom37_atom_exists",
    "atom14_gt_exists",
    "atom14_gt_positions",
    "atom14_alt_gt_positions",
    "atom14_alt_gt_exists",
    "atom14_atom_is_ambiguous",
    "rigidgroups_gt_frames",
    "rigidgroups_gt_exists",
    "rigidgroups_group_exists",
    "rigidgroups_group_is_ambiguous",
    "rigidgroups_alt_gt_frames",
    "backbone_rigid_tensor",
    "backbone_rigid_mask",
]


class StructureFeaturizer:
    def _openfold_features_from_pdb(
        self, pdb_str: str, pdb_id: T.Optional[str] = None
    ) -> OFProtein:
        """Create rigid groups from a PDB file on disk.

        The inputs to the Frame-Aligned Point Error (FAPE) loss used in AlphaFold2 are
        tuples of translations and rotations from the reference frame. In the OpenFold
        implementation, this is stored as `Rigid` objects. This function calls the
        OpenFold wrapper functions which creates an `OFProtein` object,
        and then extracts several `Rigid` objects.

        Args:
            pdb_str (str): String representing the contents of a PDB file

        Returns:
            OFProtein: _description_
        """
        pdb_id = "" if pdb_id is None else pdb_id
        protein_object = protein_from_pdb_string(pdb_str)

        # TODO: what is the `is_distillation` argument?
        protein_features = make_pdb_features(
            protein_object, description=pdb_id, is_distillation=False
        )

        return protein_features

    def _process_structure_features(
        self, features: T.Dict[str, np.ndarray], seq_len: int
    ):
        """Process feature dtypes and pad to max length."""
        for k, v in features.items():
            # Handle data types in converting from numpy to torch
            if v.dtype == np.dtype("int32"):
                features[k] = torch.from_numpy(v).long()  # int32 -> int64
            elif v.dtype == np.dtype("O"):
                features[k] = v.astype(str)[0]
            else:
                # the rest are all float32. TODO: does this be float64?
                features[k] = torch.from_numpy(v)

            # Trim or pad to a fixed length for all per-specific features
            if k in FEATURES_REQUIRING_PADDING:
                features[k] = trim_or_pad_length_first(features[k], seq_len)

            # 'seq_length' is a tensor with shape equal to the aatype array length,
            # and filled with the value of the original sequence length.
            if k == "seq_length":
                features[k] = torch.full((seq_len,), features[k][0])

        # Make the mask
        idxs = torch.arange(seq_len, dtype=torch.long)
        mask = idxs < features["seq_length"]
        features["mask"] = mask.long()

        features["aatype"] = features["aatype"].argmax(dim=-1)
        return features

    def __call__(self, pdb_str: str, seq_len: int, pdb_id: T.Optional[str] = None):
        features = self._openfold_features_from_pdb(pdb_str, pdb_id)
        features = self._process_structure_features(features, seq_len)
        features = make_all_atom_aatype(features)
        features = make_seq_mask(features)
        features = make_atom14_masks(features)
        features = make_atom14_positions(features)
        features = atom37_to_frames(features)
        features = get_backbone_frames(features)

        # f = make_pseudo_beta("")
        # p = f(p)

        # f = atom37_to_torsion_angles("")
        # p = f(p)

        # p = get_chi_angles(p)
        return features


def view_py3Dmol(pdbstr):
    import py3Dmol

    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(pdbstr)
    view.setStyle({"model": -1}, {"cartoon": {"color": "green"}})
    view.zoomTo()
    view.show()

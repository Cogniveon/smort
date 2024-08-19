import os
import logging
import h5py
from pathlib import Path
from typing import Optional
from einops import rearrange
import torch
import hydra
from omegaconf import DictConfig

import numpy as np

from smotdm.rifke import joints_to_rifke

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="configs", config_name="compute_guoh3dfeats", version_base="1.3"
)
def compute_guoh3dfeats(cfg: DictConfig):
    base_dir = cfg.base_dir
    output_file = cfg.output_file

    from smotdm.utils import loop_interx

    from smplx import SMPLX

    logger.info("Get h3d features from Guo et al.")
    logger.info(f"The processed motions will be stored in this file: {output_file}")

    iterator = loop_interx(base_dir)
    dataset = h5py.File(output_file, 'w')
    motions_dataset = dataset.create_group('motions')
    texts_dataset = dataset.create_group('texts')

    for scene_id, num_frames, motions, texts in iterator:
        # joints[..., 0] *= -1
        # joints_m = swap_left_right(joints)

        feats = []
        for motion in motions:
            smplx_model = SMPLX(
                model_path='deps/smplx/SMPLX_NEUTRAL.npz',
                batch_size=num_frames,
                num_betas=10,
                use_pca=False,
                use_face_contour=True
            )
            output = smplx_model(
                **motion,
                return_full_pose=True,
            )
            # XZY -> XYZ
            joints = torch.from_numpy(
                np.dot(output.joints.detach().cpu().numpy(), np.array(
                    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
                ))
            )

            feats.append(joints_to_rifke(joints[:, :smplx_model.NUM_JOINTS, :]))
        scene_motions = torch.stack(feats)
        assert scene_motions.shape[-1] == 163, f'Invalid feats shape({scene_id}): {scene_motions.shape}'

        logger.debug(scene_motions.shape)

        scene_dataset = motions_dataset.create_dataset(scene_id, data=scene_motions[:, :num_frames, :].detach().cpu().numpy())
        scene_dataset.attrs.create('num_frames', data=num_frames)
        texts_dataset.create_dataset(scene_id, data=np.bytes_(texts))
    
    dataset.close()
    logger.info("done.")

if __name__ == "__main__":
    compute_guoh3dfeats()

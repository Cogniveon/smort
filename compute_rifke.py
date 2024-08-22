import logging
import os
import h5py
import torch
import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from smotdm.geometry import axis_angle_rotation
from smotdm.rifke import get_forward_direction, joints_to_feats, joints_to_rifke
from smotdm.utils import compute_joints, get_smplx_model

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="compute_rifke", version_base="1.3")
def compute_rifke(cfg: DictConfig):
    base_dir = cfg.base_dir
    dataset_file = cfg.dataset_file
    device = torch.device(cfg.device)
    ids = cfg.ids
    fps = cfg.fps

    from smotdm.utils import loop_interx

    from smplx import SMPLX

    logger.info("Get h3d features from Guo et al.")
    logger.info(f"The processed motions will be stored in this file: {dataset_file}")

    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist.")
        return

    iterator = loop_interx(base_dir, device=device, include_only=ids)
    dataset = h5py.File(dataset_file, "w")
    motions_df = None

    motions_dataset = dataset.create_group("motions")
    texts_dataset = dataset.create_group("texts")

    flush_counter = 10
    for scene_id, motions, texts in iterator:
        # joints[..., 0] *= -1
        # joints_m = swap_left_right(joints)
        assert len(motions) == 2, f"Expected 2 motions, got {len(motions)}"

        reactor_motion, actor_motion = motions
        smplx_model = get_smplx_model(reactor_motion["body_pose"].shape[0], device=device)

        j1 = compute_joints(
            smplx_model,
            reactor_motion,
            device,
        )

        j2 = compute_joints(
            smplx_model,
            actor_motion,
            device,
        )
        
        translation = j1[..., 0, :].clone()
        forward = get_forward_direction(j1, jointstype="smplxjoints")
        angles = torch.atan2(*(forward.transpose(0, -1))).transpose(0, -1)

        reactor_feats = joints_to_feats(j1, translation, angles)
        actor_feats = joints_to_feats(j2, translation, angles)

        scene_motions = torch.stack(
            [
                reactor_feats,
                actor_feats,
            ]
        )
        assert (
            scene_motions.shape[-1] == 166
        ), f"Invalid feats shape({scene_id}): {scene_motions.shape}"

        num_frames = scene_motions[0].shape[0]

        motions_df = pd.concat(
            [
                *([motions_df] if motions_df is not None else []),
                pd.DataFrame(
                    [
                        [f"{scene_id}_{0}", 0, num_frames / fps],
                        [f"{scene_id}_{1}", 0, num_frames / fps],
                    ],
                    columns=["key", "start", "end"],
                ),
            ]
        )

        motions_dataset.create_dataset(
            scene_id, data=scene_motions[:, :num_frames, :].detach().cpu().numpy()
        )
        texts_dataset.create_dataset(scene_id, data=np.bytes_(texts))

        flush_counter -= 1
        if flush_counter <= 0:
            flush_counter = 100
            dataset.flush()

    assert motions_df is not None
    motions_df.to_csv(cfg.motions_path)
    dataset.close()
    logger.info("done.")


if __name__ == "__main__":
    compute_rifke()

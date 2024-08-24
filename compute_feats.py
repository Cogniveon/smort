import logging
import os
from pathlib import Path

import h5py
import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from smotdm.metrics import MeanStdMetric
from smotdm.rifke import get_forward_direction, joints_to_feats
from smotdm.utils import compute_joints, get_smplx_model, loop_interx

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="compute_feats", version_base="1.3")
def compute_feats(cfg: DictConfig):
    base_dir = cfg.base_dir
    dataset_file = cfg.dataset_file
    device = torch.device(cfg.device)
    include_only = cfg.include_only
    fps = cfg.fps
    min_seconds = cfg.min_seconds
    max_seconds = cfg.max_seconds

    Path(dataset_file).parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"The processed motions will be stored in this file: {dataset_file}")

    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist.")
        return

    iterator = loop_interx(
        base_dir,
        device=device,
        include_only=include_only,
        fps=fps,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )
    dataset = h5py.File(dataset_file, "w")
    motions_df = None
    mean_std = MeanStdMetric(166).to(device)

    motions_dataset = dataset.create_group("motions")
    texts_dataset = dataset.create_group("texts")
    text_model = instantiate(cfg.text_encoder, device=device)

    flush_counter = 10
    errored_scenes = []

    for scene_id, motions, num_frames, texts in iterator:
        try:
            assert len(motions) == 2, f"Expected 2 motions, got {len(motions)}"

            reactor_motion, actor_motion = motions
            smplx_model = get_smplx_model(
                reactor_motion["body_pose"].shape[0], device=device
            )

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

            del j1, j2, forward, translation, angles
            torch.cuda.empty_cache()

            scene_motions = torch.stack(
                [
                    reactor_feats,
                    actor_feats,
                ]
            )
            assert (
                scene_motions.shape[-1] == 166
            ), f"Invalid feats shape({scene_id}): {scene_motions.shape}"

            # Update mean and standard deviation metrics
            mean_std.update(scene_motions[0], reactor_feats.shape[0])
            mean_std.update(scene_motions[1], actor_feats.shape[0])

            motions_df = pd.concat(
                [
                    *([motions_df] if motions_df is not None else []),
                    pd.DataFrame(
                        [
                            [f"{scene_id}", 0, num_frames],
                            [f"{scene_id}", 1, num_frames],
                        ],
                        columns=["scene_id", "motion_id", "num_frames"],
                    ),
                ]
            )

            motions_dataset.create_dataset(
                scene_id, data=scene_motions[:, :num_frames, :].detach().cpu().numpy()
            )
            texts_dataset.create_dataset(
                scene_id,
                data=text_model(texts)["x"].detach().cpu().numpy(),
            )

            del scene_motions
            torch.cuda.empty_cache()

            flush_counter -= 1
            if flush_counter <= 0:
                flush_counter = 100
                dataset.flush()

        except Exception as e:
            logger.error(f"Error processing scene_id {scene_id}: {e}")
            errored_scenes.append(scene_id)
            torch.cuda.empty_cache()
            continue

    mean_value, std_value = mean_std.compute()

    stats_dataset = dataset.create_group("stats")
    stats_dataset.create_dataset("mean", data=mean_value.detach().cpu().numpy())
    stats_dataset.create_dataset("std", data=std_value.detach().cpu().numpy())

    assert motions_df is not None
    motions_df.to_csv(cfg.motions_path)
    dataset.close()

    if errored_scenes:
        logger.warning(f"Scenes with errors: {errored_scenes}")

    logger.info("done.")


if __name__ == "__main__":
    compute_feats()

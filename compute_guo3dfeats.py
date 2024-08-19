import logging
import h5py
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
    device = torch.device(cfg.device)

    from smotdm.utils import loop_interx

    from smplx import SMPLX

    logger.info("Get h3d features from Guo et al.")
    logger.info(f"The processed motions will be stored in this file: {output_file}")

    iterator = loop_interx(base_dir, device=device)
    dataset = h5py.File(output_file, "w")
    motions_dataset = dataset.create_group("motions")
    texts_dataset = dataset.create_group("texts")

    for scene_id, motions, texts in iterator:
        # joints[..., 0] *= -1
        # joints_m = swap_left_right(joints)

        feats = []
        for motion in motions:
            with torch.no_grad():
                smplx_model = SMPLX(
                    model_path="deps/smplx/SMPLX_NEUTRAL.npz",
                    num_betas=10,
                    use_pca=False,
                    use_face_contour=True,
                    batch_size=motion['body_pose'].shape[0]
                ).to(device)
                output = smplx_model(
                    **motion,
                    return_full_pose=True,
                )
                joints = torch.matmul(
                    output.joints,
                    torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device),
                )

            feats.append(joints_to_rifke(joints[:, : smplx_model.NUM_JOINTS, :]))
        scene_motions = torch.stack(feats)
        assert (
            scene_motions.shape[-1] == 163
        ), f"Invalid feats shape({scene_id}): {scene_motions.shape}"

        
        num_frames = np.min(scene_motions[:].shape[0])

        scene_dataset = motions_dataset.create_dataset(
            scene_id, data=scene_motions[:, :num_frames, :].detach().cpu().numpy()
        )
        scene_dataset.attrs.create("num_frames", data=num_frames)
        texts_dataset.create_dataset(scene_id, data=np.bytes_(texts))

    dataset.close()
    logger.info("done.")


if __name__ == "__main__":
    compute_guoh3dfeats()

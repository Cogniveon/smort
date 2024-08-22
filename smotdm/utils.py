import os
from typing import Literal, Optional
import torch
import numpy as np
from tqdm.auto import tqdm
from smplx import SMPLX

def get_smplx_model(
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    return SMPLX(
        model_path="deps/smplx/SMPLX_NEUTRAL.npz",
        num_betas=10,
        use_pca=False,
        use_face_contour=True,
        batch_size=batch_size if batch_size is not None else 1,
    ).to(device)


@torch.no_grad()
def compute_joints(
    smplx_model: SMPLX,
    smplx_params: dict,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    smplx_model.to(device)
    output = smplx_model(
        **smplx_params,
    )

    return torch.matmul(
        output.joints,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
        ),
    )[:, : smplx_model.NUM_JOINTS, :]


def loop_interx(
    base_dir: str,
    include_only: Literal["all"] | list[str] | int = "all",
    exclude: list[str] = [],
    device=torch.device("cpu"),
):
    if type(include_only) == int:
        include_only = os.listdir(f"{base_dir}/motions")[:include_only]
    pbar = tqdm(os.listdir(f"{base_dir}/motions"))
    assert type(include_only) == str or type(include_only) == list, type(include_only)
    for scene_id in pbar:
        if include_only != "all" and scene_id not in include_only:
            continue
        if scene_id in exclude:
            continue
        with open(f"{base_dir}/texts/{str(scene_id)}.txt", "r") as annotation:
            texts = []
            for line in annotation.readlines():
                texts.append(line.rstrip("\n"))

            motions = []
            for file in [
                f"{base_dir}/motions/{scene_id}/P1.npz",
                f"{base_dir}/motions/{scene_id}/P2.npz",
            ]:
                data = np.load(file)
                motions.append(
                    {
                        "body_pose": torch.tensor(
                            data["pose_body"], dtype=torch.float32, device=device
                        ),
                        "left_hand_pose": torch.tensor(
                            data["pose_lhand"], dtype=torch.float32, device=device
                        ),
                        "right_hand_pose": torch.tensor(
                            data["pose_rhand"], dtype=torch.float32, device=device
                        ),
                        "transl": torch.tensor(
                            data["trans"], dtype=torch.float32, device=device
                        ),
                        "global_orient": torch.tensor(
                            data["root_orient"], dtype=torch.float32, device=device
                        ),
                    }
                )

            pbar.set_postfix(dict(scene_id=scene_id))
            yield scene_id, motions, texts

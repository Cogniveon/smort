import math
import os
from typing import Literal, Optional
from omegaconf import ListConfig
import torch
import numpy as np
from tqdm.auto import tqdm
from smplx import SMPLX
from typing import Optional
import torch


def get_smplx_model(
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    model_path: str = "deps/smplx/SMPLX_NEUTRAL.npz",
) -> SMPLX:
    """
    Returns an instance of the SMPLX model, configured with the given batch size and device.

    Args:
        batch_size (Optional[int]): The batch size for the model. Defaults to 1 if not specified.
        device (Optional[torch.device]): The device to run the model on (CPU or GPU). Defaults to CPU.
        model_path (str): The path to the SMPLX model file. Defaults to 'deps/smplx/SMPLX_NEUTRAL.npz'.

    Returns:
        SMPLX: The configured SMPLX model instance.
    """
    # Initialize the SMPLX model
    smplx_model = SMPLX(
        model_path=model_path,
        num_betas=10,
        use_pca=False,
        use_face_contour=True,
        batch_size=batch_size if batch_size is not None else 1,
    )

    # Move the model to the specified device
    smplx_model.to(device)

    return smplx_model


@torch.no_grad()
def compute_joints(
    smplx_model: SMPLX,
    smplx_params: dict,
    device: Optional[torch.device] = torch.device("cpu"),
) -> torch.Tensor:
    smplx_model.to(device)

    output = smplx_model(**smplx_params)

    transform_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
    )
    transformed_joints = torch.matmul(output.joints, transform_matrix)

    return transformed_joints[:, : smplx_model.NUM_JOINTS, :]


def loop_interx(
    base_dir: str,
    include_only: Literal["all"] | ListConfig | list[str] | int = "all",
    exclude: list[str] = [],
    device=torch.device("cpu"),
    fps: float = 20.0,
    min_seconds: float = 2.0,
    max_seconds: float = 30.0,
):
    if include_only == "all":
        motions = os.listdir(f"{base_dir}/motions")
    elif type(include_only) == int:
        motions = os.listdir(f"{base_dir}/motions")[:include_only]
    elif type(include_only) == ListConfig:
        motions = [
            x
            for x in os.listdir(f"{base_dir}/motions")
            if x in include_only and x not in exclude
        ]
    pbar = tqdm(motions, total=len(motions))
    for scene_id in pbar:
        with open(f"{base_dir}/texts/{str(scene_id)}.txt", "r") as annotation:
            texts = []
            for line in annotation.readlines():
                texts.append(line.rstrip("\n"))

            motions = []
            end: Optional[int] = None
            for file in [
                f"{base_dir}/motions/{scene_id}/P1.npz",
                f"{base_dir}/motions/{scene_id}/P2.npz",
            ]:
                data = np.load(file)
                num_frames = data["pose_body"].shape[0]

                if num_frames < math.floor(min_seconds * fps):
                    continue
                if end is None:
                    end = math.floor(max_seconds * fps)

                motions.append(
                    {
                        "body_pose": torch.tensor(
                            data["pose_body"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "left_hand_pose": torch.tensor(
                            data["pose_lhand"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "right_hand_pose": torch.tensor(
                            data["pose_rhand"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "transl": torch.tensor(
                            data["trans"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                        "global_orient": torch.tensor(
                            data["root_orient"][0:end, ...],
                            dtype=torch.float32,
                            device=device,
                        ),
                    }
                )

            if end is None:
                continue
            pbar.set_postfix(dict(scene_id=scene_id))
            yield scene_id, motions, end, texts

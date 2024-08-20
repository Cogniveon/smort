import os
from typing import Literal
import torch
import numpy as np
from tqdm.auto import tqdm


def loop_interx(
    base_dir: str,
    include_only: Literal["all"] | list[str] = "all",
    exclude: list[str] = [],
    device=torch.device("cpu"),
):
    pbar = tqdm(os.listdir(f"{base_dir}/motions"))
    for scene_id in pbar:
        if include_only != "all" and scene_id not in include_only:
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

import torch

import pandas as pd
from torch.utils.data import Dataset

from smotdm.data.collate import collate_text_motion
from smotdm.data.loader import MotionLoader


class MotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motions_file: str,
        motion_loader: MotionLoader,
        min_seconds: float = 2.0,
        max_seconds: float = 30.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.collate_fn = collate_text_motion
        self.dataset_path = path
        self.motion_loader = motion_loader
        self.device = device
        motions = pd.read_csv(motions_file, names=["key", "start", "end"], header=0)
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # Filter the motions based on the duration
        motions["start"] = pd.to_numeric(motions["start"])
        motions["end"] = pd.to_numeric(motions["end"])
        motions["duration"] = motions["end"] - motions["start"]
        filtered_motions = motions[
            (motions["duration"] >= self.min_seconds)
            & (motions["duration"] <= self.max_seconds)
        ]
        self.motions = filtered_motions.drop(columns=["duration"]).reset_index(
            drop=True
        )
        # print(filtered_motions["duration"].min())
        # print(filtered_motions["duration"].max())
        # print(filtered_motions["duration"].mean())
        # print(filtered_motions["duration"].std())

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        motion_id = self.motions.loc[index]["key"]
        scene_id, scene_idx = motion_id.split("_")
        scene_idx = int(scene_idx)
        motion_x_dict = self.motion_loader(
            scene_id,
            scene_idx,
            self.motions.loc[index]["start"],
            self.motions.loc[index]["end"],
        )

        for key in motion_x_dict:
            if isinstance(motion_x_dict[key], torch.Tensor):
                motion_x_dict[key] = motion_x_dict[key].to(self.device)

        output = {
            "motion_x_dict": motion_x_dict,
            # "text_x_dict": text_x_dict,
            # "text": text,
            # "keyid": keyid,
            # "sent_emb": sent_emb,
        }
        return output

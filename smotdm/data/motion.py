import os
import h5py
import torch

import pandas as pd
from torch.utils.data import Dataset

from smotdm.data.collate import collate_text_motion


class MotionLoader:
    def __init__(self, dataset_file: str, fps, normalizer=None, nfeats=None):
        self.fps = fps
        self.dataset_file = dataset_file
        self.motions = {}
        self.reference_translations = {}
        self.reference_angles = {}
        self.normalizer = normalizer
        self.nfeats = nfeats
    
    def __call__(self, scene_id: str, i: int, start: float, end: float):
        begin = int(start * self.fps)
        end = int(end * self.fps)
        if scene_id not in self.motions:
            with h5py.File(self.dataset_file, "r") as f:
                motions = f[str(f"motions/{scene_id}")]
                assert type(motions) == h5py.Dataset
                motion = torch.from_numpy(motions[i]).to(torch.float)
                if self.normalizer is not None:
                    motion = self.normalizer(motions[i])
                self.motions[f"{scene_id}_{i}"] = motion

        motion = self.motions[f"{scene_id}_{i}"][begin:end]
        x_dict = {"x": motion, "length": len(motion)}
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.mean_path = os.path.join(base_dir, "mean.pt")
        self.std_path = os.path.join(base_dir, "std.pt")
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        self.mean = torch.load(self.mean_path)
        self.std = torch.load(self.std_path)

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        torch.save(mean.detach().cpu(), self.mean_path)
        torch.save(std.detach().cpu(), self.std_path)

    def __call__(self, x):
        if self.disable:
            return x
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        x = x * (self.std + self.eps) + self.mean
        return x


class MotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motions_file: str,
        motion_loader: MotionLoader,
        min_seconds: float = 2.0,
        max_seconds: float = 30.0,
        normalizer: Normalizer | None = None,
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
        motions['start'] = pd.to_numeric(motions['start'])
        motions['end'] = pd.to_numeric(motions['end'])
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

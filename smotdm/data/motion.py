import os
from random import randint
from typing import Optional
import h5py
import torch

import pandas as pd
from torch.utils.data import Dataset

from smotdm.data.collate import collate_text_motion
from smotdm.data.loader import MotionLoader

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
        self.mean = torch.load(self.mean_path, weights_only=True)
        self.std = torch.load(self.std_path, weights_only=True)

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


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.collate_fn = collate_text_motion
        self.dataset_path = path
        self.normalizer = Normalizer("deps/interx")
        self.device = device

    def __len__(self):
        with h5py.File(self.dataset_path, "r") as f:
            motions_dataset = f["motions"]
            assert type(motions_dataset) == h5py.Group
            return len(list(motions_dataset.keys()))

    def __getitem__(self, index):
        with h5py.File(self.dataset_path, "r") as f:
            motions_dataset = f["motions"]
            assert type(motions_dataset) == h5py.Group
            scene_id = list(motions_dataset.keys())[index]

            scene_dataset = motions_dataset[f"{scene_id}"]
            assert type(scene_dataset) == h5py.Dataset

            texts_dataset = f[f"texts/{scene_id}"]
            assert type(texts_dataset) == h5py.Dataset

            text = texts_dataset[randint(0, 2)]

            reactor_motion = scene_dataset[0]
            actor_motion = scene_dataset[1]
            reactor_x_dict = {
                "x": torch.tensor(
                    reactor_motion, dtype=torch.float32, device=self.device
                ),
                "length": len(reactor_motion),
            }
            actor_x_dict = {
                "x": torch.tensor(
                    actor_motion, dtype=torch.float32, device=self.device
                ),
                "length": len(actor_motion),
            }
            text_x_dict = {
                "x": torch.tensor(text, dtype=torch.float32, device=self.device),
                "length": len(text),
            }
            return {
                "reactor_x_dict": reactor_x_dict,
                "actor_x_dict": actor_x_dict,
                "text_x_dict": text_x_dict,
            }


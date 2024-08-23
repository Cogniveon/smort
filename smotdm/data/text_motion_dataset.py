from random import randint
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

from smotdm.data.collate import collate_text_motion


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_only: bool = False,
        normalize: bool = True,
        eps: float = 1e-12,
        device: torch.device = torch.device("cpu"),
    ):
        self.collate_fn = collate_text_motion
        self.dataset_path = path
        self.motion_only = motion_only
        self.normalize = normalize
        self.eps = eps
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

            reactor_motion = scene_dataset[0]
            actor_motion = scene_dataset[1]

            if self.normalize:
                mean = f["stats/mean"][()]  # type: ignore
                std = f["stats/std"][()]  # type: ignore

                assert type(mean) == np.ndarray
                assert type(std) == np.ndarray

                reactor_motion = (reactor_motion - mean) / (std + self.eps)
                actor_motion = (actor_motion - mean) / (std + self.eps)

            ret_dict = {}
            ret_dict["reactor_x_dict"] = {
                "x": torch.tensor(
                    reactor_motion, dtype=torch.float32, device=self.device
                ),
                "length": len(reactor_motion),
            }
            ret_dict["actor_x_dict"] = {
                "x": torch.tensor(
                    actor_motion, dtype=torch.float32, device=self.device
                ),
                "length": len(actor_motion),
            }

            if not self.motion_only:
                texts_dataset = f[f"texts/{scene_id}"]
                assert type(texts_dataset) == h5py.Dataset
                text = texts_dataset[randint(0, 2)]
                ret_dict["text_x_dict"] = {
                    "x": torch.tensor(text, dtype=torch.float32, device=self.device),
                    "length": len(text),
                }

            return ret_dict

    def reverse_norm(self, motion: np.ndarray):
        with h5py.File(self.dataset_path, "r") as f:
            mean = f["stats/mean"][()]  # type: ignore
            std = f["stats/std"][()]  # type: ignore

            assert type(mean) == np.ndarray
            assert type(std) == np.ndarray

            return (motion * (std + self.eps)) + mean

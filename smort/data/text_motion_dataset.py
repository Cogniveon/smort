import math
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from smort.data.collate import collate_text_motion


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_only: bool = False,
        normalize: bool = True,
        eps: float = 1e-12,
        device: torch.device = torch.device("cpu"),
        use_tiny: float = 1.0,
        return_scene: bool = False,
    ):
        self.collate_fn = collate_text_motion
        self.dataset_path = path
        self.motion_only = motion_only
        self.return_scene = return_scene
        self.normalize = normalize
        self.use_tiny = use_tiny
        self.eps = eps
        self.device = device

    def __len__(self):
        with h5py.File(self.dataset_path, "r") as f:
            motions_dataset = f["motions"]
            assert type(motions_dataset) is h5py.Group
            num_scenes = len(list(motions_dataset.keys()))
            if self.use_tiny:
                return math.floor(num_scenes * self.use_tiny)
            else:
                return num_scenes

    def get_scene(self, scene_id: str | int):
        with h5py.File(self.dataset_path, "r") as f:
            motions_dataset = f["motions"]
            assert type(motions_dataset) is h5py.Group
            if type(scene_id) == int:
                scene_id = list(motions_dataset.keys())[scene_id]
            scene_dataset = motions_dataset[f"{scene_id}"]
            assert type(scene_dataset) is h5py.Dataset

            reactor_motion = scene_dataset[0]
            actor_motion = scene_dataset[1]
            reactor_len = len(reactor_motion)
            actor_len = len(actor_motion)

            if self.normalize:
                mean = f["stats/mean"][()][: reactor_motion.shape[0], :]  # type: ignore
                std = f["stats/std"][()][: reactor_motion.shape[0], :]  # type: ignore

                assert type(mean) is np.ndarray
                assert type(std) is np.ndarray

                reactor_motion = (reactor_motion - mean) / (std + self.eps)
                actor_motion = (actor_motion - mean) / (std + self.eps)

            ret_dict = {}
            ret_dict["reactor_x_dict"] = {
                "x": torch.tensor(
                    reactor_motion, dtype=torch.float32, device=self.device
                ),
                "length": reactor_len,
            }
            ret_dict["actor_x_dict"] = {
                "x": torch.tensor(
                    actor_motion, dtype=torch.float32, device=self.device
                ),
                "length": actor_len,
            }

            if not self.motion_only:
                texts_dataset = f[f"texts/{scene_id}"]
                assert type(texts_dataset) is h5py.Dataset
                text = random.choice(texts_dataset[()])
                ret_dict["text_x_dict"] = {
                    "x": torch.tensor(text, dtype=torch.float32, device=self.device),
                    "length": len(text),
                }

            if self.return_scene:
                ret_dict["scene_x_dict"] = {}
                ret_dict["scene_x_dict"]["x"] = torch.cat(
                    (
                        ret_dict["reactor_x_dict"]["x"],
                        ret_dict["actor_x_dict"]["x"],
                    ),
                    dim=1,
                )
                assert (
                    ret_dict["actor_x_dict"]["length"]
                    == ret_dict["actor_x_dict"]["length"]
                )
                ret_dict["scene_x_dict"]["length"] = ret_dict["reactor_x_dict"][
                    "length"
                ]
                # del ret_dict['reactor_x_dict']
                # del ret_dict['actor_x_dict']
            return ret_dict

    def __getitem__(self, index):
        return self.get_scene(index)

    def get_mean_std(
        self, return_tensors: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.dataset_path, "r") as f:
            mean = f["stats/mean"][()]  # type: ignore
            std = f["stats/std"][()]  # type: ignore
            if return_tensors:
                return torch.from_numpy(mean), torch.from_numpy(std) + self.eps
            assert type(mean) == np.ndarray and type(std) == np.ndarray
            return mean, std + self.eps

    def reverse_norm(
        self, motion: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if type(motion) is torch.Tensor:
            motion = motion.detach().cpu().numpy()
        assert type(motion) == np.ndarray
        with h5py.File(self.dataset_path, "r") as f:
            mean = f["stats/mean"][()][: motion.shape[0], :]  # type: ignore
            std = f["stats/std"][()][: motion.shape[0], :]  # type: ignore

            assert type(mean) is np.ndarray
            assert type(std) is np.ndarray

            return (motion * (std + self.eps)) + mean

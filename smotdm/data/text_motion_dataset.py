from random import randint
import h5py
import torch

from torch.utils.data import Dataset

from smotdm.data.collate import collate_text_motion
from smotdm.data.normalizer import Normalizer


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


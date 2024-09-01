import math
import os
from pathlib import Path

import boto3
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from smort.data.text_motion_dataset import TextMotionDataset


class InterXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_file: str,
        batch_size=32,
        num_workers=os.cpu_count(),
        use_tiny: float = 1.0,
        return_scene: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_file = dataset_file
        self.batch_size = batch_size
        self.num_workers = num_workers or 1
        self.use_tiny = use_tiny
        self.return_scene = return_scene

    def prepare_data(self):
        if not os.path.exists(self.dataset_file):
            client = boto3.client("s3")
            client.download_file(
                "eeem004-storage", Path(self.dataset_file).name, self.dataset_file
            )

    def setup(self, stage: str) -> None:
        self.prepare_data()
        self.dataset = TextMotionDataset(
            self.dataset_file, use_tiny=self.use_tiny, return_scene=self.return_scene
        )

        total_scenes = len(self.dataset)
        train_len = math.floor(total_scenes * 0.8)
        val_test_len = (total_scenes - train_len) // 2

        self.train, self.val, self.test = random_split(
            self.dataset, [train_len, val_test_len, val_test_len]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            collate_fn=self.dataset.collate_fn,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            collate_fn=self.dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test,
            collate_fn=self.dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def get_scene(self, scene_id: str | int):
        return self.dataset.get_scene(scene_id)

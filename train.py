import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from smotdm.config import read_config, save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    config_path = save_config(cfg)
    logger.info(f"The config can be found here: {config_path}")

    import pytorch_lightning as pl
    from torch.utils.data import DataLoader

    pl.seed_everything(cfg.seed)

    logger.info("Loading dataloaders")
    train_dataset = instantiate(cfg.data)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=os.cpu_count() or 1,
        persistent_workers=True,
    )

    logger.info("Loading model")
    model = instantiate(cfg.model, dataset=train_dataset)

    logger.info("Training")
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader)
    logger.info("Training done.")
    

if __name__ == "__main__":
    train()

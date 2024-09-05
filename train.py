import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from smort.config import save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    import pytorch_lightning as pl

    from smort.data.data_module import InterXDataModule
    from smort.models.smort import SMORT
    
    config_path = save_config(cfg)
    logger.info(f"The config can be found here: {config_path}")

    pl.seed_everything(cfg.seed)

    logger.info("Loading datamodule")
    data_module: InterXDataModule = instantiate(cfg.data)
    data_module.setup("fit")

    logger.info("Loading model")
    mean, std = data_module.dataset.get_mean_std()
    logger.info("Loading trainer")
    trainer: pl.Trainer = instantiate(cfg.trainer)
    
    if not cfg.resume_from_ckpt:
        model: SMORT = instantiate(cfg.model, data_mean=mean, data_std=std)
    else:
        try:
            from pytorch_lightning.loggers.wandb import WandbLogger
            if type(trainer.logger) == WandbLogger:
                model_id = Path(cfg.resume_from_ckpt).parent.name
                trainer.logger.download_artifact(f"rohit-k-kesavan/smort/{model_id}", artifact_type="model")
        except Exception as e:
            logger.error(e)

        model = SMORT.load_from_checkpoint(cfg.resume_from_ckpt, data_mean=mean, data_std=std)

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    logger.info("Training done.")


if __name__ == "__main__":
    train()

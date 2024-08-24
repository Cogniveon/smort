import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from smotdm.config import save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    config_path = save_config(cfg)
    logger.info(f"The config can be found here: {config_path}")

    import pytorch_lightning as pl
    from smotdm.data.data_module import InterXDataModule
    from smotdm.models.smotdm import SMOTDM

    pl.seed_everything(cfg.seed)

    logger.info("Loading datamodule")
    data_module: InterXDataModule = instantiate(cfg.data)
    data_module.setup("train")

    logger.info("Loading model")
    mean, std = data_module.dataset.get_mean_std()
    # model: SMOTDM = instantiate(cfg.model, data_mean=mean, data_std=std)
    model = SMOTDM.load_from_checkpoint('local_model.ckpt', data_mean=mean, data_std=std)

    logger.info("Training")
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.fit(model, data_module)
    logger.info("Training done.")


if __name__ == "__main__":
    train()

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
    from torch.utils.data import DataLoader

    pl.seed_everything(cfg.seed)

    logger.info("Loading datamodule")
    data_module = instantiate(cfg.data)
    data_module.setup("train")

    logger.info("Loading model")
    model = instantiate(cfg.model, dataset=data_module.dataset)

    logger.info("Training")
    trainer: pl.Trainer = instantiate(cfg.trainer)
    trainer.fit(model, data_module)
    logger.info("Training done.")


if __name__ == "__main__":
    train()

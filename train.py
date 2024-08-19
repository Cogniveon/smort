import logging
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

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")

    # train_dataset = instantiate(cfg.data, split="train")
    # val_dataset = instantiate(cfg.data, split="val")

    # train_dataloader = instantiate(
    #     cfg.dataloader,
    #     dataset=train_dataset,
    #     collate_fn=train_dataset.collate_fn,
    #     shuffle=True,
    # )

    # val_dataloader = instantiate(
    #     cfg.dataloader,
    #     dataset=val_dataset,
    #     collate_fn=val_dataset.collate_fn,
    #     shuffle=False,
    # )

    # logger.info("Loading the model")
    # model = instantiate(cfg.model)

    # logger.info("Training")
    # trainer = instantiate(cfg.trainer)
    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()

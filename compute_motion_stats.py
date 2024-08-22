import logging
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from smotdm.data.motion import MotionDataset, MotionLoader, Normalizer
from smotdm.data.collate import collate_text_motion

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="configs", config_name="compute_motion_stats", version_base="1.3"
)
def compute_motion_stats(cfg: DictConfig):
    motions_file = cfg.motions_file
    dataset_file = cfg.dataset_file
    batch_size = cfg.batch_size
    device = torch.device(cfg.device)

    dataset = MotionDataset(
        dataset_file, motions_file, MotionLoader(dataset_file, 20.0), None, device=device
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_text_motion, num_workers=4
    )
    all_feats = []
    for batch in tqdm(loader):
        all_feats.append(torch.cat([x for x in batch["motion_x_dict"]["x"]]))

    feats = torch.cat(all_feats)
    mean = feats.mean(dim=0)
    std = feats.std(dim=0)

    normalizer = Normalizer("deps/interx", disable=True)
    logger.info(f"Saving them in {normalizer.base_dir}")
    normalizer.save(mean, std)

    logger.info("done.")


if __name__ == "__main__":
    compute_motion_stats()

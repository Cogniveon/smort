import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from smort.config import save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="visualize", version_base="1.3")
def visualize(cfg: DictConfig):
    config_path = save_config(cfg)
    logger.info(f"The config can be found here: {config_path}")

    import pytorch_lightning as pl
    from smort.data.data_module import InterXDataModule
    from smort.models.smort import SMORT
    from smort.renderer.matplotlib import SceneRenderer
    from smort.rifke import feats_to_joints
    import torch

    pl.seed_everything(cfg.seed)

    logger.info("Loading datamodule")
    data_module: InterXDataModule = instantiate(cfg.data)
    data_module.setup("train")

    logger.info("Loading model")
    mean, std = data_module.dataset.get_mean_std()
    if cfg.ckpt:
        model = SMORT.load_from_checkpoint(cfg.ckpt, data_mean=mean, data_std=std)
    else:
        model: SMORT = instantiate(cfg.model, data_mean=mean, data_std=std)
    
    sample = data_module.get_scene(cfg.input)
    sample = data_module.dataset.collate_fn([sample])
    
    if cfg.infer_type == 'actor':
        s_motions, s_latents, s_dists = model(
            sample["actor_x_dict"],
            "actor",
            mask=sample['reactor_x_dict']['mask'],
            return_all=True,
        )
        
        renderer = SceneRenderer()
        
        pred_motion = data_module.dataset.reverse_norm(s_motions[0])
        pred_joints = feats_to_joints(torch.from_numpy(pred_motion))
        
        gt_motion = data_module.dataset.reverse_norm(sample["reactor_x_dict"]['x'][0])
        gt_joints = feats_to_joints(torch.from_numpy(gt_motion))
        
        actor_motion = data_module.dataset.reverse_norm(sample["actor_x_dict"]['x'][0])
        actor_joints = feats_to_joints(torch.from_numpy(actor_motion))
        
        renderer.render_animation([pred_joints, actor_joints], output="viz.mp4")
        renderer.render_animation([gt_joints, actor_joints], output="gt.mp4")


if __name__ == "__main__":
    visualize()
import logging
import random
from typing import Dict, List, Literal, Optional

import torch
from pytorch_lightning import LightningModule
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from smort.data.collate import length_to_mask
from smort.models.losses import JointLoss, KLLoss
from smort.models.modules import ACTORStyleDecoder, ACTORStyleEncoder
from smort.renderer.matplotlib import SceneRenderer
from smort.rifke import feats_to_joints

logger = logging.getLogger(__name__)


class SMORT(LightningModule):
    def __init__(
        self,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        nmotionfeats: int = 76,
        ntextfeats: int = 768,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        vae: bool = True,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {
            "recons": 1,
            "joint": 1.0e-2,
            "latent_s": 1.0e-5,
            "latent_t": 1.0e-5,
            "kl": 1.0e-4,
        },
        lr: float = 1e-4,
        viz_interval: int = -1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_mean", "data_std"])

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

        self.motion_encoder = ACTORStyleEncoder(
            nfeats=nmotionfeats,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )
        self.scene_encoder = ACTORStyleEncoder(
            nfeats=nmotionfeats * 2,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )
        self.text_encoder = ACTORStyleEncoder(
            nfeats=ntextfeats,
            vae=vae,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )
        self.motion_decoder = ACTORStyleDecoder(
            nfeats=nmotionfeats,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
        )

        self.data_mean = data_mean
        self.data_std = data_std

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()
        self.joint_loss_fn = JointLoss(self.data_mean, self.data_std)

        # lambda weighting for the losses
        self.lmd = lmd
        self.lr = lr

        self.renderer = SceneRenderer()
        self.viz_interval = viz_interval

    def configure_optimizers(self):
        optimizer = AdamW(
            lr=self.lr,
            params=self.parameters(),
            weight_decay=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(optimizer, mode="min"),
            "monitor": "val_loss",
        }

    def forward(
        self,
        inputs: Dict,
        input_type: Literal["motion", "scene", "text"],
        context: Dict,
        lengths: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        if input_type == "text":
            encoder = self.text_encoder
        elif input_type == "scene":
            encoder = self.scene_encoder
        else:
            encoder = self.motion_encoder

        encoded = encoder(inputs)

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        # import pdb; pdb.set_trace()
        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.motion_decoder(z_dict, context)

        if return_all:
            return motions, latent_vectors, dists

        return motions

    def compute_loss(
        self,
        batch: Dict,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        ref_motions = batch["reactor_x_dict"]["x"]
        mask = batch["reactor_x_dict"]["mask"]

        # scene -> motion
        s_motions, s_latents, s_dists = self(
            batch["scene_x_dict"],
            "scene",
            mask=mask,
            context=batch["actor_x_dict"],
            return_all=True,
        )
        # text -> motion
        t_motions, t_latents, t_dists = self(
            batch["text_x_dict"],
            "text",
            mask=mask,
            context=batch["actor_x_dict"],
            return_all=True,
        )
        # motion -> motion
        m_motions, m_latents, m_dists = self(
            batch["reactor_x_dict"],
            "motion",
            mask=mask,
            context=batch["actor_x_dict"],
            return_all=True,
        )

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(s_motions, ref_motions) # scene -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # motion -> motion
        )
        losses["joint"] = (
            + self.joint_loss_fn.forward(t_motions, ref_motions, mask)
            + self.joint_loss_fn.forward(s_motions, ref_motions, mask)
            + self.joint_loss_fn.forward(m_motions, ref_motions, mask)
        )
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                +self.kl_loss_fn(t_dists, ref_dists)  # text
                + self.kl_loss_fn(s_dists, ref_dists)  # scene
                + self.kl_loss_fn(ref_dists, s_dists)
                + self.kl_loss_fn(ref_dists, t_dists)
            )

        # Latent manifold loss
        losses["latent_s"] = self.latent_loss_fn(s_latents, m_latents)
        losses["latent_t"] = self.latent_loss_fn(t_latents, m_latents)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        return losses, m_motions, ref_motions

    def viz_random_from_batch(
        self,
        motion: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        output: str = "viz.mp4",
    ):
        bs = motion.shape[0]
        randidx = random.randint(0, bs - 1)

        pred_joints, gt_joints = (
            feats_to_joints(
                self.joint_loss_fn.denorm(motion[randidx][mask[randidx], ...])
            ),
            feats_to_joints(self.joint_loss_fn.denorm(gt[randidx][mask[randidx], ...])),
        )

        self.renderer.render_animation(
            [pred_joints, gt_joints],
            output=output,
        )

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])

        # self.lmd = {
        #     **self.lmd,
        #     "joint": cosine_annealing_lambda(current_epoch % 100, 100, 1.0, 2) * 1e-2,
        # }

        losses, pred_motions, gt_motions = self.compute_loss(batch)
        assert type(losses) is dict

        # metrics = self.joint_loss_fn.evaluate_metrics(
        #     pred_motions, gt_motions, batch["reactor_x_dict"]["mask"]
        # )
        # for metric_name in sorted(metrics):
        #     metric_val = metrics[metric_name]
        #     self.log(
        #         f"train_{metric_name}",
        #         metric_val,
        #         on_epoch=True,
        #         on_step=True,
        #         batch_size=bs,
        #     )

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
                prog_bar=loss_name == "loss",
            )

        if (
            self.viz_interval != -1
            and (
                (self.trainer.current_epoch) % self.viz_interval == 0
                or self.trainer.current_epoch + 1 == self.trainer.max_epochs
            )
            and batch_idx == 0
        ):
            self.viz_random_from_batch(
                pred_motions,
                gt_motions,
                batch["reactor_x_dict"]["mask"],
                output="local_train_viz.mp4",
            )

        if batch_idx == 0:
            loss_dict = {k: v.item() for k, v in losses.items()}
            # metrics_dict = {k: v.item() for k, v in metrics.items()}
            # - Metrics dict: {metrics_dict}
            logger.info(
                f"Train Epoch {self.trainer.current_epoch} - Loss dict: {loss_dict}"
            )

        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])

        losses, pred_motions, gt_motions = self.compute_loss(batch)
        assert type(losses) is dict

        metrics = self.joint_loss_fn.evaluate_metrics(
            pred_motions, gt_motions, batch["reactor_x_dict"]["mask"]
        )

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        for metric_name in sorted(metrics):
            metric_val = metrics[metric_name]
            self.log(
                f"val_{metric_name}",
                metric_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        if self.viz_interval != -1 and batch_idx == 0:
            self.viz_random_from_batch(
                pred_motions,
                gt_motions,
                batch["reactor_x_dict"]["mask"],
                output="local_val_viz.mp4",
            )

        # import pdb; pdb.set_trace()
        if batch_idx == 0:
            loss_dict = {k: v.item() for k, v in losses.items()}
            metrics_dict = {k: v.item() for k, v in metrics.items()}
            logger.info(
                f"Val Epoch {self.trainer.current_epoch} - Metrics dict: {metrics_dict} - Loss dict: {loss_dict}"
            )
        return losses["loss"]

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])

        losses, pred_motions, gt_motions = self.compute_loss(batch)
        assert type(losses) is dict

        metrics = self.joint_loss_fn.evaluate_metrics(
            pred_motions, gt_motions, batch["reactor_x_dict"]["mask"]
        )

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"test_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        for metric_name in sorted(metrics):
            metric_val = metrics[metric_name]
            self.log(
                f"test_{metric_name}",
                metric_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        return losses["loss"]

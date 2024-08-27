import random
from typing import Dict, List, Literal, Optional

import torch
from pytorch_lightning import LightningModule
from torch.optim.adamw import AdamW

from smort.data.collate import length_to_mask
from smort.models.losses import JointLoss, KLLoss
from smort.models.modules import (
    ACTORStyleDecoder,
    ACTORStyleEncoder,
    ACTORStyleEncoderWithCA,
)
from smort.renderer.matplotlib import SceneRenderer
from smort.rifke import feats_to_joints


def cosine_annealing_lambda(epoch: int, num_epochs: int, initial_lambda: float) -> torch.Tensor:
    """
    Returns a lambda value based on a cosine annealing schedule.
    :param epoch: Current epoch.
    :param num_epochs: Total number of epochs.
    :param initial_lambda: Initial value of lambda.
    :return: Adjusted lambda value.
    """
    return initial_lambda * 0.5 * (1 + torch.cos(torch.tensor(epoch * 3.14159265 / num_epochs)))


class SMORT(LightningModule):
    def __init__(
        self,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        nmotionfeats: int = 166,
        ntextfeats: int = 768,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        dropout: float = 0.1,
        activation: str = "gelu",
        vae: bool = True,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1, "joint": 1.0e-5, "latent": 1.0e-5, "kl": 1.0e-5},
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

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
        self.text_encoder = ACTORStyleEncoderWithCA(
            nfeats=ntextfeats,
            n_context_feats=nmotionfeats,
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

    def configure_optimizers(self):
        return {
            "optimizer": AdamW(lr=self.lr, params=self.parameters()),
        }

    def forward(
        self,
        inputs: Dict,
        input_type: Literal["scene", "text"],
        context: Optional[Dict] = None,
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
            encoder = self.reactor_encoder

        if isinstance(encoder, ACTORStyleEncoderWithCA):
            assert context is not None
            encoded = encoder(inputs, context)
        else:
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
        motions = self.motion_decoder(z_dict)

        if return_all:
            return motions, latent_vectors, dists

        return motions

    def compute_loss(
        self,
        batch: Dict,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        # actor_x_dict = batch["actor_x_dict"]
        ref_motions = batch["reactor_x_dict"]["x"]

        # encoder types: "reactor", "actor"
        # # text -> motion
        # t_motions, t_latents, t_dists = self(
        #     text_x_dict, "text", mask=mask, return_all=True
        # )
        # actor -> motion
        t_motions, t_latents, t_dists = self(
            batch["text_x_dict"], "text", context=batch["actor_x_dict"], mask=batch["reactor_x_dict"]['mask'], return_all=True
        )
        # scene -> motion
        m_motions, m_latents, m_dists = self(
            batch["scene_x_dict"], "scene", mask=batch["reactor_x_dict"]['mask'], return_all=True
        )

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # scene -> motion
        )
        # fmt: on

        losses["joint"] = self.joint_loss_fn.forward(t_motions, ref_motions, batch["reactor_x_dict"]['mask'])

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                +self.kl_loss_fn(t_dists, ref_dists)  # text
                + self.kl_loss_fn(m_dists, ref_dists)  # scene
                + self.kl_loss_fn(ref_dists, m_dists)
                + self.kl_loss_fn(ref_dists, t_dists)
            )

        # Latent manifold loss
        losses["latent_a"] = self.latent_loss_fn(t_latents, m_latents)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        return losses, m_motions, ref_motions

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])
        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs or 100
        
        # Update lambda values using cosine annealing schedule
        recons_lambda = cosine_annealing_lambda(current_epoch, max_epochs, self.lmd["recons"])
        joint_lambda = cosine_annealing_lambda(current_epoch, max_epochs, self.lmd["joint"])

        # Apply the updated lambda values
        self.lmd["recons"] = recons_lambda
        self.lmd["joint"] = joint_lambda

        losses, pred_motions, gt_motions = self.compute_loss(batch)
        # import pdb; pdb.set_trace()
        # if batch_idx == 0:
        #     randidx = random.randint(0, bs - 1)
        #     pred_joints, gt_joints = (
        #         self.joint_loss_fn._denorm_and_to_joints(pred_motions[randidx]),
        #         self.joint_loss_fn._denorm_and_to_joints(gt_motions[randidx]),
        #     )
        #     self.render_motion(pred_joints, gt_joints, "local_train_viz.mp4")
        assert type(losses) is dict

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
                prog_bar=loss_name == "loss" or loss_name == "joint",
            )
        return losses["loss"]

    def render_motion(self, motion: torch.Tensor, gt: torch.Tensor, output=None):
        if output is None:
            assert self.logger is not None
            output = "viz.mp4"

        self.renderer.render_animation(
            gt.detach().cpu().numpy(),
            motion.detach().cpu().numpy(),
            output=output,
        )

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])

        losses, pred_motions, gt_motions = self.compute_loss(batch)
        # import pdb; pdb.set_trace()
        if batch_idx == 0:
            randidx = random.randint(0, bs - 1)
            pred_joints, gt_joints = (
                self.joint_loss_fn.to_joints(pred_motions[randidx]),
                self.joint_loss_fn.to_joints(gt_motions[randidx]),
            )
            self.render_motion(pred_joints, gt_joints, "local_val_viz.mp4")
        assert type(losses) is dict
        # if batch_idx == 0:
        #     random_idx = random.randint(0, bs - 1)
        #     # import pdb; pdb.set_trace()
        #     self.render_motion(joints[random_idx], gt_joints[random_idx], "local_val_viz.mp4")

        # for metric_name in sorted(metrics):
        #     loss_val = metrics[metric_name]
        #     self.log(
        #         f"val_{metric_name}",
        #         loss_val,
        #         on_epoch=True,
        #         on_step=True,
        #         batch_size=bs,
        #     )

        # for metric_name in sorted(losses):
        #     loss_val = losses[metric_name]
        #     self.log(
        #         f"val_{metric_name}",
        #         loss_val,
        #         on_epoch=True,
        #         on_step=True,
        #         batch_size=bs,
        #     )
        return losses["loss"]

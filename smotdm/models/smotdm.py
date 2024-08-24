from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.optim.adamw import AdamW

from smotdm.data.collate import length_to_mask
from smotdm.models.losses import KLLoss
from smotdm.models.modules import ACTORStyleDecoder, ACTORStyleEncoder
from smotdm.models.text_encoder import TextToEmb
from smotdm.renderer.matplotlib import SingleMotionRenderer
from smotdm.rifke import feats_to_joints


class SMOTDM(LightningModule):
    def __init__(
        self,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        nmotionfeats: int = 166,
        ntextfeats: int = 768,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        vae: bool = True,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5},
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

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

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # lambda weighting for the losses
        self.lmd = lmd
        self.lr = lr

        self.data_mean = data_mean
        self.data_std = data_std
        self.renderer = SingleMotionRenderer(
            colors=("red", "red", "red", "red", "red"),
        )

    def configure_optimizers(self):
        return {
            "optimizer": AdamW(lr=self.lr, params=self.parameters()),
        }

    def forward(
        self,
        inputs,
        input_type: Literal["reactor", "text"],
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
        # elif input_type == "actor":
        #     encoder = self.actor_encoder
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

        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        z_dict = {"z": latent_vectors, "mask": mask}
        motions = self.motion_decoder(z_dict)

        if return_all:
            return motions, latent_vectors, dists

        return motions

    def compute_loss(self, batch: Dict) -> Dict:
        text_x_dict = batch["text_x_dict"]
        actor_x_dict = batch["actor_x_dict"]
        reactor_x_dict = batch["reactor_x_dict"]

        mask = reactor_x_dict["mask"]
        ref_motions = reactor_x_dict["x"]

        # text -> motion
        t_motions, t_latents, t_dists = self(
            text_x_dict, "text", mask=mask, return_all=True
        )
        # # actor -> reactor motion
        # a_motions, a_latents, a_dists = self(
        #     actor_x_dict, "actor", mask=mask, return_all=True
        # )
        # motion -> motion
        m_motions, m_latents, m_dists = self(
            reactor_x_dict, "reactor", mask=mask, return_all=True
        )

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(m_motions, ref_motions) # reactor -> motion
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
                self.kl_loss_fn(t_dists, m_dists)  # text_to_motion
                + self.kl_loss_fn(m_dists, t_dists)  # motion_to_text
                + self.kl_loss_fn(m_dists, ref_dists)  # motion
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent_t"] = self.latent_loss_fn(t_latents, m_latents)
        # losses["latent_a"] = self.latent_loss_fn(a_latents, m_latents)

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )
        return losses

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        bs = len(batch["reactor_x_dict"]["x"])
        losses = self.compute_loss(batch)

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
        return losses["loss"]
    
    def norm_and_render_motion(self, motion: torch.Tensor, output=None):
        if output is None:
            assert self.logger is not None
            output = "viz.mp4"
        if self.data_std.device != motion.device:
            self.data_mean.to(motion.device)
            self.data_std.to(motion.device)
        
        motion = motion * (self.data_std[:motion.shape[0], :] + 1e-12) + self.data_mean[:motion.shape[0], :]
        
        self.renderer.render_animation_single(
            feats_to_joints(motion).detach().cpu().numpy(),
            output=output
        )

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        if batch_idx == 0:
            viz_batch = batch['reactor_x_dict']
            viz_batch['x'] = viz_batch['x'][:1, ...]
            viz_batch['mask'] = viz_batch['mask'][:1, ...]
            motions, latents, dists = self(
                viz_batch, "reactor", mask=viz_batch["mask"], return_all=True
            )
            self.norm_and_render_motion(motions[0], 'viz.mp4')
            
        bs = len(batch["reactor_x_dict"]["x"])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

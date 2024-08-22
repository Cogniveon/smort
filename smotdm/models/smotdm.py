from typing import Dict, List, Literal, Optional

from pytorch_lightning import LightningModule
import torch
from torch.optim.adamw import AdamW
from smotdm.data.collate import length_to_mask
from smotdm.models.losses import KLLoss
from smotdm.models.modules import ACTORStyleDecoder, ACTORStyleEncoder


class SMOTDM(LightningModule):
    def __init__(
        self,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5},
        lr: float = 1e-4,
    ) -> None:
        super().__init__()

        self.motion_encoder = ACTORStyleEncoder(
            nfeats=166,
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
        )
        self.text_encoder = ACTORStyleEncoder(
            nfeats=768,
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
        )
        self.actor_encoder = ACTORStyleEncoder(
            nfeats=166,
            vae=True,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
        )
        self.motion_decoder = ACTORStyleDecoder(
            nfeats=166,
            latent_dim=256,
            ff_size=1024,
            num_layers=6,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
        )

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

        # losses
        self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # lambda weighting for the losses
        self.lmd = lmd
        self.lr = lr

    def configure_optimizers(self):
        return {"optimizer": AdamW(lr=self.lr, params=self.parameters())}

    def forward(
        self,
        inputs,
        input_type: Literal["reactor", "text", "actor"],
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
        elif input_type == "actor":
            encoder = self.actor_encoder
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
        a_motions, a_latents, a_dists = self(
            actor_x_dict, "actor", mask=mask, return_all=True
        )
        m_motions, m_latents, m_dists = self(
            reactor_x_dict, "reactor", mask=mask, return_all=True
        )

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off
        losses["recons"] = (
            + self.reconstruction_loss_fn(t_motions, ref_motions) # text -> motion
            + self.reconstruction_loss_fn(a_motions, ref_motions) # actor -> motion
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
                + self.kl_loss_fn(m_dists, t_dists)  # reactor_to_text
                + self.kl_loss_fn(a_dists, t_dists)  # actor_to_text
                + self.kl_loss_fn(t_dists, a_dists)  # text_to_actor
                + self.kl_loss_fn(a_dists, m_dists)  # actor_to_reactor
                + self.kl_loss_fn(m_dists, a_dists)  # reactor_to_actor
                + self.kl_loss_fn(a_dists, ref_dists)  # actor
                + self.kl_loss_fn(m_dists, ref_dists)  # reactor
                + self.kl_loss_fn(t_dists, ref_dists)  # text
            )

        # Latent manifold loss
        losses["latent_t"] = self.latent_loss_fn(t_latents, m_latents)
        losses["latent_a"] = self.latent_loss_fn(a_latents, m_latents)

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

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
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

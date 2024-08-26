import torch
import torch.nn as nn
import torch.nn.functional as F

from smort.joints import SMPLX_JOINT_PAIRS
from smort.rifke import feats_to_joints


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class JointLoss(nn.Module):
    def __init__(
        self,
        data_mean: torch.Tensor,
        data_std: torch.Tensor,
        use_joint_position_loss: bool = True,
        use_bone_length_loss: bool = False,
        use_foot_sliding_loss: bool = False,
        foot_indices: list = [10, 11],
        contact_threshold: float = 0.1,
        lmb: dict = {"joint_position": 1.0, "bone_length": 1.0, "foot_sliding": 1.0},
    ) -> None:
        super(JointLoss, self).__init__()
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        self.use_joint_position_loss = use_joint_position_loss
        self.use_bone_length_loss = use_bone_length_loss
        self.use_foot_sliding_loss = use_foot_sliding_loss
        self.foot_indices = foot_indices
        self.contact_threshold = contact_threshold
        self.lmb = lmb

    def _denorm_and_to_joints(self, motion: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        return feats_to_joints(
            motion * self.data_std[: motion.shape[-2], :]
            + self.data_mean[: motion.shape[-2], :],
        )

    @staticmethod
    def compute_bone_lengths(joints: torch.Tensor) -> torch.Tensor:
        # Using broadcasting to compute the lengths between joint pairs
        joints1 = joints[..., [pair[0] for pair in SMPLX_JOINT_PAIRS], :]
        joints2 = joints[..., [pair[1] for pair in SMPLX_JOINT_PAIRS], :]
        return torch.norm(joints1 - joints2, dim=-1)

    def compute_foot_sliding_loss(self, joints: torch.Tensor) -> torch.Tensor:
        # Compute foot velocities (difference between consecutive frames)
        velocities = joints[:, 1:, :, :] - joints[:, :-1, :, :]

        # Identify frames where the foot should be in contact (based on vertical velocity)
        contact_mask = (
            torch.abs(velocities[:, :, self.foot_indices, 2]) < self.contact_threshold
        ).float()

        # Compute foot sliding as the horizontal velocity when the foot should be in contact
        sliding_velocities = velocities[
            :, :, self.foot_indices, :2
        ]  # Consider only x and y axis
        sliding_loss = torch.mean(contact_mask * torch.norm(sliding_velocities, dim=-1))

        return sliding_loss

    def compute_mpjme(
        self,
        predicted_joints: torch.Tensor,
        gt_joints: torch.Tensor,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            movement_error = torch.norm(
                (predicted_joints[:, 1:, :, :] - predicted_joints[:, :-1, :, :])
                - (gt_joints[:, 1:, :, :] - gt_joints[:, :-1, :, :]),
                dim=-1,
            )
            mpjme = movement_error.mean()
        return mpjme

    def compute_mrpe(
        self, predicted_joints: torch.Tensor, gt_joints: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            position_error = torch.norm(
                predicted_joints[:, :, 0, :] - gt_joints[:, :, 0, :], dim=-1
            )
            mrpe = position_error.mean()
        return mrpe

    def evaluate_metrics(
        self, predicted_joints: torch.Tensor, gt_joints: torch.Tensor
    ) -> dict:
        """Compute evaluation metrics like MPJME."""
        metrics = {}
        metrics["mpjme"] = self.compute_mpjme(predicted_joints, gt_joints)
        metrics["mrpe"] = self.compute_mrpe(predicted_joints, gt_joints)
        return metrics

    def forward(
        self,
        predicted_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        mask: torch.Tensor,
        return_joints: bool = False,
    ) -> torch.Tensor:
        # Denormalize and convert to joints
        predicted_joints = self._denorm_and_to_joints(predicted_motion)
        gt_joints = self._denorm_and_to_joints(gt_motion)

        # Initialize loss
        loss = torch.tensor(0.0, dtype=torch.float32, device=predicted_motion.device)

        # Joint Position Loss
        if self.use_joint_position_loss:
            # import pdb; pdb.set_trace()
            expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, predicted_joints.size(-2), predicted_joints.size(-1))
            masked_predicted_joints = predicted_joints[expanded_mask].view(-1, 54, 3)
            masked_gt_joints = gt_joints[expanded_mask].view(-1, 54, 3)
            joint_position_loss = F.mse_loss(masked_predicted_joints, masked_gt_joints)
            loss += self.lmb["joint_position"] * joint_position_loss

        # Bone Length Loss
        if self.use_bone_length_loss:
            predicted_lengths = self.compute_bone_lengths(predicted_joints)
            gt_lengths = self.compute_bone_lengths(gt_joints)
            bone_length_loss = F.mse_loss(predicted_lengths, gt_lengths)
            loss += self.lmb["bone_length"] * bone_length_loss

        # Foot Sliding Loss
        if self.use_foot_sliding_loss:
            foot_sliding_loss = self.compute_foot_sliding_loss(predicted_joints)
            loss += self.lmb["foot_sliding"] * foot_sliding_loss

        # import pdb; pdb.set_trace()
        if return_joints:
            return loss, predicted_joints, gt_joints  # type: ignore

        return loss

    def __repr__(self):
        return "JointLoss()"

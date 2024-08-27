import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ) -> None:
        super(JointLoss, self).__init__()
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

    def to_joints(self, motion: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        return feats_to_joints(
            motion * self.data_std[: motion.shape[-2], :]
            + self.data_mean[: motion.shape[-2], :],
        )

    def get_root_position_loss(
        self, pred_joints: torch.Tensor, gt_joints: torch.Tensor
    ):
        # Trajectory
        loss = F.mse_loss(
            pred_joints[..., 0, [0, 1]], gt_joints[..., 0, [0, 1]], reduction="mean"
        )
        # height
        loss += F.mse_loss(
            pred_joints[..., 0, [2]], gt_joints[..., 0, [2]], reduction="mean"
        )
        return loss

    def get_foot_contact_loss(
        self,
        pred_joints: torch.Tensor,
        gt_joints: torch.Tensor,
        feet_indices: list = [7, 8, 10, 11],
        ground_threshold=0.01,
    ):
        # Ensure that feet joints are close to the ground (y-coordinate close to 0)
        # For simplicity, we assume ground level is at y < 0.01
        relevant_pred_joints = pred_joints[:, feet_indices, :]
        relevant_gt_joints = gt_joints[:, feet_indices, :]

        pred_vel = relevant_pred_joints[1:, :, :] - relevant_pred_joints[:-1, :, :]
        gt_vel = relevant_gt_joints[1:, :, :] - relevant_gt_joints[:-1, :, :]
        gt_vel_norm = torch.linalg.norm(gt_vel, dim=2)

        fc_mask = (gt_vel_norm <= ground_threshold).unsqueeze(2).expand(-1, -1, 3)
        masked_pred_vel = pred_vel.clone()
        masked_pred_vel[~fc_mask] = 0

        return F.mse_loss(
            masked_pred_vel, torch.zeros_like(masked_pred_vel), reduction="mean"
        )

    def forward(self, motion: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
        bs, _, nfeats = motion.shape
        loss = torch.tensor(0.0, dtype=torch.float32, device=motion.device)
        # Select some important joints
        joint_selection = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 18, 19, 20, 21, 15]

        for i in range(bs):
            pred_joints = self.to_joints(motion[i][mask[i], ...])
            gt_joints = self.to_joints(gt[i][mask[i], ...])

            loss += self.get_root_position_loss(pred_joints, gt_joints)
            loss += self.get_foot_contact_loss(pred_joints, gt_joints)
            loss += F.mse_loss(
                pred_joints[:, joint_selection, :],
                gt_joints[:, joint_selection, :],
                reduction="mean",
            )

        return loss / bs

    def compute_mpjme(
        self,
        predicted_joints: torch.Tensor,
        gt_joints: torch.Tensor,
    ) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            movement_error = torch.norm(
                (predicted_joints[:, 1:, :] - predicted_joints[:, :-1, :])
                - (gt_joints[:, 1:, :] - gt_joints[:, :-1, :]),
                dim=-1,
            )
            mpjme = movement_error.mean()
        return mpjme

    def compute_mrpe(
        self, predicted_joints: torch.Tensor, gt_joints: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            position_error = torch.norm(
                predicted_joints[:, 0, :] - gt_joints[:, 0, :], dim=-1
            )
            mrpe = position_error.mean()
        return mrpe

    def evaluate_metrics(
        self,
        predicted_motion: torch.Tensor,
        gt_motion: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """Compute evaluation metrics like MPJME."""
        metrics = {
            "mpjme": torch.tensor(
                0.0, dtype=torch.float32, device=predicted_motion.device
            ),
            "mrpe": torch.tensor(
                0.0, dtype=torch.float32, device=predicted_motion.device
            ),
        }
        bs, _, _ = predicted_motion.shape

        for i in range(bs):
            pred_joints = self.to_joints(predicted_motion[i][mask[i], ...])
            gt_joints = self.to_joints(gt_motion[i][mask[i], ...])

            metrics["mpjme"] += self.compute_mpjme(pred_joints, gt_joints)
            metrics["mrpe"] += self.compute_mrpe(pred_joints, gt_joints)

        metrics["mpjme"] /= bs
        metrics["mrpe"] /= bs
        return metrics

    # def forward(
    #     self,
    # ) -> torch.Tensor:
    #     # expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, predicted_joints.size(-2), predicted_joints.size(-1))

    #     # Denormalize and convert to joints
    #     predicted_joints = self._denorm_and_to_joints(predicted_motion)
    #     gt_joints = self._denorm_and_to_joints(gt_motion)

    #     # Initialize loss
    #     loss = torch.tensor(0.0, dtype=torch.float32, device=predicted_motion.device)

    #     expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, predicted_joints.size(-2), predicted_joints.size(-1))
    #     masked_predicted_joints = predicted_joints[expanded_mask].view(-1, 54, 3)
    #     masked_gt_joints = gt_joints[expanded_mask].view(-1, 54, 3)

    #     # root position
    #     loss += F.mse_loss(masked_predicted_joints[:, 0, :], masked_gt_joints[:, 0, :])
    #     # Joint Position Loss
    #     if self.use_joint_position_loss:
    #         # import pdb; pdb.set_trace()
    #         joint_position_loss = F.mse_loss(masked_predicted_joints, masked_gt_joints)
    #         loss += self.lmb["joint_position"] * joint_position_loss

    #     # Bone Length Loss
    #     if self.use_bone_length_loss:
    #         predicted_lengths = self.compute_bone_lengths(masked_predicted_joints)
    #         gt_lengths = self.compute_bone_lengths(masked_gt_joints)
    #         bone_length_loss = F.mse_loss(predicted_lengths, gt_lengths)
    #         loss += self.lmb["bone_length"] * bone_length_loss

    #     # Foot Sliding Loss
    #     if self.use_foot_sliding_loss:
    #         foot_sliding_loss = self.compute_foot_sliding_loss(predicted_joints)
    #         loss += self.lmb["foot_sliding"] * foot_sliding_loss

    #     # import pdb; pdb.set_trace()
    #     if return_joints:
    #         return loss, predicted_joints, gt_joints  # type: ignore

    #     return loss

    def __repr__(self):
        return "JointLoss()"

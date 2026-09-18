import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class SapiensNormalLoss(nn.Module):
    """L1 + outlier-gate normal loss (pixel3dmm / SAM-3D style).

    Both tensors must already be in the same convention ([-1,1] per channel).
    `mask` is a binary (0/1) float tensor marking pixels to include.
    """
    def __init__(self, delta: float = 0.33):
        super().__init__()
        self.delta = delta

    def forward(self, rendered: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # rendered / target: (1, 3, H, W);  mask: (1, 1, H, W)
        diff = (rendered - target) * mask
        inlier = (diff.abs().sum(dim=1, keepdim=True) / 3.0 < self.delta).float()
        denom = mask.sum().clamp(min=1.0)
        return (diff.abs() * inlier).sum() / (denom * 3.0)


class UniDepthMapLoss(nn.Module):
    """L1 + outlier-gate dense scene-depth loss (SapiensNormalLoss style, but
    for metric depth in meters instead of unit normal vectors).

    `rendered` is the mesh's rasterized camera-space Z per pixel; `target` is
    UniDepth's per-pixel scene depth (same camera, same metric convention, so
    no coordinate conversion is needed — unlike the normal loss). `mask`
    additionally excludes invalid (<=0) UniDepth pixels.
    """
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta

    def forward(self, rendered: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # rendered / target: (1, 1, H, W);  mask: (1, 1, H, W)
        diff = (rendered - target) * mask
        inlier = (diff.abs() < self.delta).float()
        denom = mask.sum().clamp(min=1.0)
        return (diff.abs() * inlier).sum() / denom


def gmof(residual, sigma):
    """
    Geman-McClure robust error function.
    """
    x_squared = residual**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

class ShapeAttributeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, est_shape, target_shape_attr):
        if len(target_shape_attr) == 0:
            return torch.zeros(est_shape.shape[0], device=est_shape.device)
        residuals = est_shape - target_shape_attr
        loss = (residuals**2).mean()
        return loss

class RelativeDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, est_depth, target_depth, conf=None):
        if len(target_depth) == 0:
            return torch.zeros(est_depth.shape[0], device=est_depth.device)
        if conf == None:
            conf = torch.ones_like(est_depth)
        residuals = (est_depth - target_depth) * conf
        loss = (residuals**2)
        return loss

class ReprojectionLoss(nn.Module):
    def __init__(self, sigma, min_conf=0.0):
        super().__init__()
        self.sigma = sigma
        self.min_conf = min_conf

    def forward(self, est_kpts, target_kpts, conf=None):
        residual = est_kpts - target_kpts
        if conf is not None:
            conf = torch.clamp(conf, min=0.0)
            residual = residual * conf.unsqueeze(-1)
        loss = gmof(residual.norm(dim=-1), self.sigma)
        if conf is not None:
            valid_mask = conf >= self.min_conf
            loss = (loss * valid_mask).mean(dim=-1)
        if loss.numel() == 0:
            return torch.zeros(est_kpts.shape[0], device=est_kpts.device)
        return loss

class PiecewiseRelativeDepthLoss(nn.Module):
    """
    Piecewise depth ordering loss (BEV-style).
    https://github.com/Arthur151/ROMP/blob/a8558aed480af850756f84e2a7c787e359bddbd0/romp/lib/loss_funcs/relative_loss.py#L46
    """
    def __init__(self, dist_thresh=0.05, equality_threshold=0.1):
        super().__init__()
        self.dist_thresh = dist_thresh
        self.equality_threshold = equality_threshold

    def forward(self, pred_depths, gt_depths):
        if pred_depths.numel() < 2:
            return torch.tensor(0.0, device=pred_depths.device)

        pred_diff = pred_depths.unsqueeze(1) - pred_depths.unsqueeze(0)
        gt_diff = gt_depths.unsqueeze(1) - gt_depths.unsqueeze(0)
        triu_mask = torch.triu(torch.ones_like(gt_diff), diagonal=1).bool()

        eq_mask = (torch.abs(gt_diff) <= self.equality_threshold) & triu_mask
        loss_equal = (pred_diff[eq_mask]**2)

        cd_mask_base = (gt_diff < -self.equality_threshold) & triu_mask
        cd_mask = cd_mask_base & ((pred_diff - gt_diff * self.dist_thresh) > 0)
        loss_closer = F.softplus(pred_diff[cd_mask])

        fd_mask_base = (gt_diff > self.equality_threshold) & triu_mask
        fd_mask = fd_mask_base & ((pred_diff - gt_diff * self.dist_thresh) < 0)
        loss_further = F.softplus(-pred_diff[fd_mask])

        total_loss = torch.cat([loss_equal, loss_closer, loss_further])
        if total_loss.numel() == 0:
            return torch.tensor(0.0, device=pred_depths.device)
        return total_loss.mean()


class BodyFittingLoss(nn.Module):
    def __init__(self, loss_cfg: DictConfig):
        super().__init__()
        self.kpts_2d_loss = ReprojectionLoss(sigma=loss_cfg.keypoints_2d.sigma, min_conf=loss_cfg.keypoints_2d.get('min_conf', 0.5))
        self.dense_kp_loss = ReprojectionLoss(sigma=loss_cfg.dense_kp.sigma, min_conf=0.0)
        self.shape_attr_loss = ShapeAttributeLoss()
        self.depth_loss = RelativeDepthLoss()
        self.kp_depth_loss = RelativeDepthLoss()
        self.ordering_loss = PiecewiseRelativeDepthLoss()

        self.kpts_2d_weight = loss_cfg.keypoints_2d.weight
        self.dense_kp_weight = loss_cfg.dense_kp.weight
        self.shape_attr_weight = loss_cfg.shape_attribute.weight
        self.depth_weight = loss_cfg.depth_weight
        self.kp_depth_weight = loss_cfg.kp_depth_weight
        self.ordering_depth_weight = loss_cfg.ordering_depth_weight

        self.shape_init_weight = loss_cfg.shape_init_weight
        self.pose_init_weight = loss_cfg.pose_init_weight
        self.verts_init_weight = loss_cfg.verts_init_weight

        normal_cfg = loss_cfg.get('normal', None)
        self.normal_weight = normal_cfg.weight if normal_cfg else 0.0
        self.normal_loss = SapiensNormalLoss(delta=normal_cfg.get('delta', 0.33) if normal_cfg else 0.33)

        depth_map_cfg = loss_cfg.get('depth_map', None)
        self.depth_map_weight = depth_map_cfg.weight if depth_map_cfg else 0.0
        self.depth_map_loss = UniDepthMapLoss(delta=depth_map_cfg.get('delta', 0.05) if depth_map_cfg else 0.05)

    def update_weights(self, stage_loss_weights: DictConfig):
        """Updates the loss weights for the current stage."""
        weights_to_update = (
            'pose_init_weight', 'shape_init_weight', 'verts_init_weight',
            'depth_weight', 'kp_depth_weight', 'ordering_depth_weight',
            'normal_weight', 'depth_map_weight',
        )
        for attr_name in weights_to_update:
            default_value = getattr(self, attr_name)
            new_value = getattr(stage_loss_weights, attr_name, default_value)
            setattr(self, attr_name, new_value)

    def forward(self, model_verts, body_pose, shape, est_kpts_2d, est_dense_kp,
                verts_init, init_pose, init_shape, target_kpts_2d, target_dense_kp,
                est_shape_attr=None, est_depth=None, est_kp_depth=None,
                est_depth_scale=None, est_depth_shift=None,
                target_shape_attr=None, target_depth=None, target_kp_depth=None,
                rendered_normals=None, target_normals=None, normal_mask=None,
                rendered_depth_map=None, target_depth_map=None, depth_map_mask=None):
        # --- Reprojection Losses ---
        loss_kpts = self.kpts_2d_loss(est_kpts_2d, target_kpts_2d[:, :, :2], conf=target_kpts_2d[:, :, 2])
        weighted_loss_kpts = self.kpts_2d_weight * loss_kpts

        # --- Regularization and Prior Losses ---
        verts_init_loss = ((model_verts - verts_init)**2).sum(dim=[-1, -2])
        pose_loss = ((body_pose - init_pose)**2).sum(dim=[-1, -2])
        beta_loss = ((shape - init_shape)**2).sum(dim=-1)

        weighted_verts_init_loss = self.verts_init_weight * verts_init_loss
        weighted_pose_loss = self.pose_init_weight * pose_loss
        weighted_shape_loss = self.shape_init_weight * beta_loss

        total_loss = (
            weighted_loss_kpts +
            weighted_verts_init_loss + weighted_pose_loss +
            weighted_shape_loss)

        loss_dict = {
            'loss/keypoints_2d': weighted_loss_kpts.detach().mean().item(),
            'loss/shape_init': weighted_shape_loss.detach().mean().item(),
            'loss/verts_init': weighted_verts_init_loss.detach().mean().item(),
            'loss/pose_init': weighted_pose_loss.detach().mean().item(),
            'loss/total_loss_per_person': total_loss.detach(),
        }

        # Dense kp: only compute when weight > 0 (placeholder zeros → div-by-zero otherwise)
        if self.dense_kp_weight > 0:
            loss_dense = self.dense_kp_loss(est_dense_kp, target_dense_kp[:, :, :2],
                                            conf=target_dense_kp[:, :, 2])
            weighted_loss_dense = self.dense_kp_weight * loss_dense
            total_loss = total_loss + weighted_loss_dense
            loss_dict['loss/dense_kp'] = weighted_loss_dense.detach().mean().item()

        total_loss = total_loss.mean()

        # --- Shape Attribute Loss ---
        if self.shape_attr_weight > 0:
            loss_shape_attr = self.shape_attr_loss(est_shape_attr, target_shape_attr)
            weighted_shape_attr_loss = self.shape_attr_weight * loss_shape_attr
            total_loss += weighted_shape_attr_loss
            loss_dict['loss/shape_attr'] = weighted_shape_attr_loss.item()

        # --- Depth Loss ---
        if self.depth_weight > 0:
            loss_depth = self.depth_loss(est_depth, target_depth)
            weighted_depth_loss = self.depth_weight * loss_depth
            total_loss += weighted_depth_loss
            loss_dict['loss/depth'] = weighted_depth_loss.item()

        if self.kp_depth_weight > 0:
            loss_kp_depth = self.kp_depth_loss(est_kp_depth, target_kp_depth[:, :, 0], conf=target_kp_depth[:, :, 1])
            weighted_kp_depth_loss = self.kp_depth_weight * loss_kp_depth
            total_loss += weighted_kp_depth_loss
            loss_dict['loss/kp_depth'] = weighted_kp_depth_loss.item()

        if self.ordering_depth_weight > 0:
            loss_ordering = self.ordering_loss(est_depth, target_depth)
            weighted_ordering_loss = self.ordering_depth_weight * loss_ordering
            total_loss += weighted_ordering_loss
            loss_dict['loss/ordering_depth'] = weighted_ordering_loss.item()

        # --- Surface Normal Loss ---
        if self.normal_weight > 0 and rendered_normals is not None:
            loss_normal = self.normal_loss(rendered_normals, target_normals, normal_mask)
            weighted_normal_loss = self.normal_weight * loss_normal
            total_loss = total_loss + weighted_normal_loss
            loss_dict['loss/normal'] = weighted_normal_loss.item()

        # --- UniDepth Dense Depth-Map Loss ---
        if self.depth_map_weight > 0 and rendered_depth_map is not None:
            loss_depth_map = self.depth_map_loss(rendered_depth_map, target_depth_map, depth_map_mask)
            weighted_depth_map_loss = self.depth_map_weight * loss_depth_map
            total_loss = total_loss + weighted_depth_map_loss
            loss_dict['loss/depth_map'] = weighted_depth_map_loss.item()

        loss_dict['loss/total_loss'] = total_loss.item()
        return total_loss, loss_dict

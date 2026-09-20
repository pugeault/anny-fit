
import os
import torch
import torch.nn.functional as F
import cv2
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from anny_wrapper import Anny, MyParameterDict
from utils import perspective_projection, get_camera_parameters, solvePnP
from render import visualize_and_save, visualize_points
from losses import BodyFittingLoss


from PIL import Image, ImageOps
import numpy as np

# ── nvdiffrast normal rendering helpers ───────────────────────────────────────

def _compute_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Differentiable area-weighted vertex normals.
    verts (B,V,3) camera-space, faces (F,3) long → unit normals (B,V,3).

    Uses non-in-place scatter_add so autograd can backprop through the normal
    accumulation into vertex positions (in-place scatter_add_ breaks the graph).
    """
    B, V, _ = verts.shape
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, F, 3)
    # Replicate fn for each of the 3 face vertices, then scatter in one call.
    # faces.reshape(-1) interleaves: [v0_f0, v1_f0, v2_f0, v0_f1, ...]
    # so fn_rep must also repeat each face normal 3× consecutively (not [fn,fn,fn]).
    faces_flat = faces.reshape(-1)                       # (3F,)
    fn_rep     = fn.repeat_interleave(3, dim=1)          # (B, 3F, 3): fn0,fn0,fn0, fn1,fn1,fn1…
    idx        = faces_flat.view(1, -1, 1).expand(B, -1, 3)
    vn = torch.zeros(B, V, 3, device=verts.device, dtype=verts.dtype)
    vn = vn.scatter_add(1, idx, fn_rep)                 # non-in-place: gradient flows
    return F.normalize(vn, dim=-1)


def _verts_to_clip(verts_cam: torch.Tensor, K: torch.Tensor,
                   W: int, H: int, near: float = 0.01, far: float = 100.0) -> torch.Tensor:
    """Project camera-space verts (B,V,3) to nvdiffrast clip coords (B,V,4).
    K shape (1,3,3). Uses exact pinhole formula — works for any cx, cy."""
    fx, fy = K[0, 0, 0], K[0, 1, 1]
    cx, cy = K[0, 0, 2], K[0, 1, 2]
    x, y, z = verts_cam[..., 0], verts_cam[..., 1], verts_cam[..., 2]
    # nvdiffrast image convention: NDC y=-1 → row 0 (top), y=+1 → row H-1 (bottom).
    # Camera space Y is also down (OpenCV), so no y-flip needed here.
    xc = (2.0 * fx * x + (2.0 * cx - W) * z) / W
    yc = (2.0 * fy * y + (2.0 * cy - H) * z) / H
    # Linear depth (near→-1, far→+1)
    zc = (2.0 * z / (far - near) - (far + near) / (far - near)) * z
    wc = z.clone()
    return torch.stack([xc, yc, zc, wc], dim=-1).contiguous()

class AnnyfitStage(pl.LightningModule):
    """
    AnnyfitStage: Optimizes an Anny model instance by fitting it to 2D keypoints.
    """
    def __init__(self, target, cfg: DictConfig, initial_params: dict,  img_path: str, logger, K: torch.Tensor=None):
        super().__init__()
        # saves the config and creates self.hparams
        self.cfg = cfg
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        self.save_hyperparameters(cfg_dict)
        # init the model
        self.num_people = initial_params['root_translation_params'].shape[0]
        self.anny_model = Anny(batch_size=self.num_people)
        self.init_target(target)
        self.anny_model.init_parameters(initial_params)

        self.img_path = img_path
        self.img_name = os.path.basename(img_path)
        self.exp_logger = logger
        self.vis_dir = os.path.join(self.exp_logger.log_dir, 'vis')
        os.makedirs(self.vis_dir, exist_ok=True)

        self.img = cv2.imread(self.img_path)
        img_size = self.img.shape[:2] # (height, width)

        if K is None:
            print("Camera matrix K not provided. Initializing with default values.")
            K = get_camera_parameters(
                img_size=img_size,
                fov=self.cfg.camera.fov,
                p_x=0.5,
                p_y=0.5
            )
            K = K.unsqueeze(0)  # add batch dimension

        self.faces = self.anny_model.faces

        self.register_buffer('K', K)

        self.fitting_loss = BodyFittingLoss(loss_cfg=self.cfg.loss)

        # ── nvdiffrast normal renderer ─────────────────────────────────────────
        # SAPIENS normals are 1280×720; render at that resolution by default,
        # or smaller if cfg.render is set (the target is resized to match via
        # F.interpolate in _render_scene() either way) — nvdiffrast cost scales
        # with pixel count, so this is a direct lever on per-iteration cost.
        render_cfg = self.cfg.get('render', None)
        self._rnd_W = render_cfg.width if render_cfg else 1280
        self._rnd_H = render_cfg.height if render_cfg else 720
        src_H, src_W = img_size          # full-image dims from cv2 (H, W)
        K_rnd = K.clone()
        K_rnd[..., 0, :] *= self._rnd_W / src_W  # scale fx, cx
        K_rnd[..., 1, :] *= self._rnd_H / src_H  # scale fy, cy
        self.register_buffer('K_rnd', K_rnd)

        # int32 faces for nvdiffrast
        self.register_buffer('faces_int', self.faces.int())

        self._glctx = None  # lazy init on first forward (after .to(device))
        self._waist_joint_idx = None  # lazy lookup on first _render_scene() call

    def init_target(self, target):
        self.target = TargetData()
        self.target.set_data('keypoints_2d', target['keypoints_2d'])
        if 'dense_kp' in target:
            self.target.set_data('dense_kp', target['dense_kp'])

        if 'depth' in target:
            self.target.set_data('depth', target['depth'])

        if 'keypoints_2d_depth' in target:
            self.target.set_data('keypoints_2d_depth', target['keypoints_2d_depth'])

        if 'shape_attributes' in target:
            shape_attr = target['shape_attributes']
            indices = shape_attr.get('attr_indices', [])
            if len(indices) > 0:
                self.target.set_data('shape_attr_batch_indices', torch.tensor(shape_attr['batch_indices'], dtype=torch.long))
                self.target.set_data('shape_attr_indices', torch.tensor(indices, dtype=torch.long))
                self.target.set_data('shape_attr_values', torch.tensor(shape_attr['attr_values'], dtype=torch.float32))

        if 'masks' in target:
            self.target.set_data('masks', target['masks'])

        if 'depth_map' in target:
            self.target.set_data('depth_map', target['depth_map'])

        if 'sapiens_normal' in target:
            self.target.set_data('sapiens_normal', target['sapiens_normal'])

    def visualize_mesh(self, vertices, save_path, keypoints=None):
        vis_img = self.img.copy()
        try:
            if keypoints is not None:
                vis_img = visualize_points(vis_img, keypoints, save_path, color=(0, 0, 255))
                # add target points
                vis_img = visualize_points(vis_img, self.target.keypoints_2d[:, :, :2].clone().detach().cpu(), save_path, color=(0, 255, 0))
        except:
            print("Error visualizing keypoints.")
        visualize_and_save(vis_img, vertices, self.faces, self.K, save_path)

    def _render_scene(self, verts_cam, all_joints_3d, need_normal: bool, need_depth: bool):
        """Rasterize the mesh once and read off surface normals and/or camera-space
        depth per pixel, sharing a single rasterization pass between both losses.

        Returns
        -------
        rendered_normal  (1, 3, H, W) or None  — rendered normals in SAPIENS convention
        target_normal    (1, 3, H, W) or None  — SAPIENS normal resized to render resolution
        normal_mask      (1, 1, H, W) or None  — fg_mask ∩ upper-body mask
        rendered_depth   (1, 1, H, W) or None  — rendered camera-space Z (metric, meters)
        target_depth     (1, 1, H, W) or None  — UniDepth scene depth resized to render resolution
        depth_mask       (1, 1, H, W) or None  — fg_mask ∩ valid target-depth mask
        """
        import nvdiffrast.torch as dr

        W, H = self._rnd_W, self._rnd_H
        device = verts_cam.device

        # Lazy context init (must happen after model is on GPU)
        if self._glctx is None:
            self._glctx = dr.RasterizeCudaContext(device=device)

        # Concat all people's vertices for joint rasterization
        # verts_cam: (B, V, 3)
        B, V, _ = verts_cam.shape

        attrs = []
        if need_normal:
            # Compute vertex normals in our camera space (Y-down, Z-forward)
            # faces_int is a registered buffer (on the correct device)
            vn = _compute_vertex_normals(verts_cam, self.faces_int.long())  # (B, V, 3)
            # Convert to SAPIENS convention: X→X, Y→-Y, Z→-Z
            # (OpenCV Y-down/Z-forward → OpenGL-like Y-up/Z-toward-viewer used by SAPIENS)
            vn_sap = vn * vn.new_tensor([1.0, -1.0, -1.0])  # (B, V, 3)
            attrs.append(vn_sap)
        if need_depth:
            attrs.append(verts_cam[..., 2:3])  # (B, V, 1) camera-space Z, metric — same
                                                # convention/units as UniDepth's depth_map

        attr_cat = torch.cat(attrs, dim=-1)  # (B, V, 3), (B, V, 1) or (B, V, 4)

        # Build clip-space vertices (combine all people into one mesh)
        verts_clip = _verts_to_clip(verts_cam, self.K_rnd, W, H)  # (B, V, 4)

        # Rasterize (nvdiffrast expects a single batch with offset faces)
        # For B=1 the logic is straightforward; for B>1 we render each person separately
        # and union the fg masks. Here B is typically 1.
        interp_list, fg_list = [], []
        for b in range(B):
            v_b = verts_clip[b:b+1]                           # (1, V, 4)
            a_b = attr_cat[b:b+1]                             # (1, V, C)
            rast, _ = dr.rasterize(self._glctx, v_b, self.faces_int, [H, W])
            interp, _ = dr.interpolate(a_b, rast, self.faces_int)  # (1, H, W, C)
            fg = (rast[..., 3] > 0).float().unsqueeze(-1)       # (1, H, W, 1)
            interp_list.append(interp)
            fg_list.append(fg)

        interp_img = torch.stack([r[0] for r in interp_list], dim=0).mean(dim=0, keepdim=True)  # (1,H,W,C)
        fg_mask = torch.stack([f[0] for f in fg_list], dim=0).max(dim=0).values.unsqueeze(0)  # (1,H,W,1)
        fg_mask = fg_mask.permute(0, 3, 1, 2)  # (1,1,H,W)

        c = 0
        rendered_normal = target_normal = normal_mask = None
        rendered_depth = target_depth = depth_mask = None

        if need_normal:
            nrm_img = interp_img[..., c:c + 3]; c += 3
            # → (1, 3, H, W)
            nrm_img = F.normalize(nrm_img.permute(0, 3, 1, 2), dim=1)  # (1,3,H,W)

            # Waist mask: keep pixels above the waist-level spine joint's y-coordinate.
            # all_joints_3d indexing is [root, *body_pose_keys] (163 = 1 + 162), and
            # this rig's spine chain runs spine01 (upper torso, near the chest) down
            # to spine05 (lowest, essentially at the pelvis) — spine03 sits roughly
            # midway, at anatomical waist height. (The previous version used indices
            # [1, 2], intended as "left/right hip" per its comment, but those are
            # actually pelvis.L and upperleg01.L — two LEFT-side joints, not
            # left+right — and they land at pelvis height, well below the waist.)
            if self._waist_joint_idx is None:
                names = ['root'] + self.anny_model.get_bodypose_parameters()[0]
                self._waist_joint_idx = names.index('spine03')
            waist_3d = all_joints_3d[:, self._waist_joint_idx:self._waist_joint_idx + 1, :]  # (B,1,3)
            waist_2d = perspective_projection(waist_3d, self.K_rnd)          # (B,1,2)
            waist_y_px = waist_2d[:, :, 1].max().item()                      # scalar: waist y in px
            y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1).expand(1, 1, H, W)
            upper_mask = (y_grid <= waist_y_px).float()                      # (1,1,H,W)

            normal_mask = fg_mask * upper_mask                          # (1,1,H,W)

            # Resize SAPIENS target to render resolution if needed
            tgt = self.target.sapiens_normal                              # (1,3,H_s,W_s)
            if tgt.shape[-2] != H or tgt.shape[-1] != W:
                tgt = F.interpolate(tgt, size=(H, W), mode='bilinear', align_corners=False)
            target_normal = tgt
            rendered_normal = nrm_img

        if need_depth:
            rendered_depth = interp_img[..., c:c + 1].permute(0, 3, 1, 2)  # (1,1,H,W)

            tgt_d = self.target.depth_map.float()
            if tgt_d.dim() == 2:
                tgt_d = tgt_d.unsqueeze(0).unsqueeze(0)  # (1,1,H_s,W_s)
            if tgt_d.shape[-2] != H or tgt_d.shape[-1] != W:
                tgt_d = F.interpolate(tgt_d, size=(H, W), mode='bilinear', align_corners=False)
            target_depth = tgt_d

            # UniDepth pixels with no valid reading are 0 — exclude from the loss.
            valid = (tgt_d > 0).float()
            depth_mask = fg_mask * valid

        return rendered_normal, target_normal, normal_mask, rendered_depth, target_depth, depth_mask

    def get_ignore_params(self, stage):
        ignore_params = []
        if "fingertoe" in stage.ignore_params:
            ignore_params += list(self.anny_model.fingertoe_params_names)
        if "face" in stage.ignore_params:
            ignore_params += list(self.anny_model.face_params_names)
        return ignore_params

    def _unpack_parameters(self, param_container, ignore_params=None):
        if ignore_params is None:
            ignore_params = set()

        # case 1: custom container with nested parameters
        if isinstance(param_container, MyParameterDict):
            return [p for name, p in param_container.items() if name not in ignore_params]

        # case 2: a PyTorch ParameterDict
        if isinstance(param_container, torch.nn.ParameterDict):
            return list(param_container.values())

        # case 3: a single parameter tensor
        if isinstance(param_container, torch.nn.Parameter):
            return [param_container]

        return []

    def init_optimizer(self, stage):
        """Creates a new optimizer for a given stage."""
        # freeze all parameters to reset state
        for param in self.parameters():
            param.requires_grad = False

        params_for_stage = []
        lr_config = self.cfg.optimizer.param_groups

        ignore_params = self.get_ignore_params(stage)

        for param_name in stage.params:
            param_container = getattr(self.anny_model, param_name)

            # get individual tensors from nested parameter containers
            unpacked_params = self._unpack_parameters(param_container, ignore_params)

            # unfreeze the unpacked parameters
            for p in unpacked_params:
                p.requires_grad = True

            # assign learning rate for the group
            lr_key = f"{param_name}_lr"
            lr = lr_config.get(lr_key, self.cfg.optimizer.lr)
            params_for_stage.append({'params': unpacked_params, 'lr': lr})

        if not params_for_stage:
            raise ValueError(f"Optimizer has no parameters for stage. Check stage config: {stage.params}")

        return torch.optim.Adam(params_for_stage)

    def log_step(self, losses, global_step, stage_idx, stage):
        if not hasattr(self.exp_logger, 'experiment'):
            return
        metrics_to_log = losses.copy()
        metrics_to_log['stage'] = float(stage_idx)
        for key, value in metrics_to_log.items():
            if isinstance(value, torch.Tensor):
                continue
            self.exp_logger.experiment.add_scalar(key, value, global_step=global_step)

    def get_final_params(self):
        pose_keys, pose_values = self.anny_model.get_bodypose_parameters()
        shape_keys, shape_values = self.anny_model.get_shape_parameters()
        local_keys, local_values = self.anny_model.get_local_changes_parameters()
        final_params = {
            'root_rotation_params': self.anny_model.root_rotation_params.clone().detach().unsqueeze(1), # (bs, 1, 3)
            'root_translation_params': self.anny_model.root_translation_params.clone().detach(), # (bs, 3)
            'body_pose_params': pose_values.clone().detach(), # (bs, 162, 3)
            'shape_params': shape_values.clone().detach(), # (bs, 11)
            'local_changes_params': local_values.clone().detach(), # (bs, 256)
            'body_pose_keys': [pose_keys] * self.num_people,  # (bs, 162)
            'shape_keys': [shape_keys] * self.num_people,  # (bs, 11)
            'local_changes_keys': [local_keys] * self.num_people,  # (bs, 256)
            'camera_intrinsics': self.K.clone().detach().repeat(self.num_people, 1, 1),  # (bs, 3, 3)
        }

        return final_params

    def get_final_vertices(self):
        return self.anny_model.verts_init.clone().detach()  # (bs, V, 3)

    def get_initial_vertices(self):
        return self.initial_vertices  # (bs, V, 3)

    def _perform_one_optimization_step(self, optimizer, step_idx):
        optimizer.zero_grad()
        # forward through the body model
        anny_output = self.anny_model.forward()

        # project 3D points to 2D
        coco_joints = anny_output['coco_joints']  # (bs, 163, 3)
        est_kpts_2d = perspective_projection(coco_joints, self.K)
        # Only project dense joints when their loss is active (placeholder zeros → div-by-zero)
        if self.fitting_loss.dense_kp_weight > 0:
            est_dense_kpts_2d = perspective_projection(anny_output['dense_joints'], self.K)
        else:
            est_dense_kpts_2d = anny_output['dense_joints'].new_zeros(
                anny_output['dense_joints'].shape[0], anny_output['dense_joints'].shape[1], 2)

        # estimate the shape attributes
        est_shape_attr = anny_output['shape'][self.target.shape_attr_batch_indices, self.target.shape_attr_indices]

        est_depth_scale = anny_output['depth_scale']
        est_depth_shift = anny_output['depth_shift']

        # ── normal / depth-map rendering (shared rasterization pass) ───────────
        rendered_normals = target_normals = normal_mask = None
        rendered_depth_map = target_depth_map = depth_map_mask = None
        need_normal = (self.fitting_loss.normal_weight > 0
                      and self.target.sapiens_normal.numel() > 0)
        need_depth = (self.fitting_loss.depth_map_weight > 0
                     and self.target.depth_map.numel() > 0)
        if need_normal or need_depth:
            (rendered_normals, target_normals, normal_mask,
             rendered_depth_map, target_depth_map, depth_map_mask) = \
                self._render_scene(anny_output['vertices'], coco_joints,
                                   need_normal=need_normal, need_depth=need_depth)
            if need_depth:
                # Same depth_scale/depth_shift affine calibration used by the sparse
                # depth losses (anny_output['scaled_depth']): AnnyFIT's internal Z is
                # not guaranteed to already be metric, so it must be learned jointly
                # (add 'depth_params' to the stage's params list) rather than assumed.
                rendered_depth_map = (rendered_depth_map * anny_output['depth_scale']
                                      + anny_output['depth_shift'])

        # Calculate losses
        total_loss, loss_dict = self.fitting_loss(
            model_verts=anny_output['vertices'],
            body_pose=anny_output['body_pose'],
            shape=anny_output['shape'],
            est_kpts_2d=est_kpts_2d,
            est_dense_kp=est_dense_kpts_2d,
            verts_init=self.anny_model.verts_init,
            init_pose=self.anny_model.init_pose,
            init_shape=self.anny_model.init_shape,
            target_kpts_2d=self.target.keypoints_2d,
            target_dense_kp=self.target.dense_kp,
            est_shape_attr=est_shape_attr,
            est_depth=anny_output['scaled_depth'],
            est_kp_depth=anny_output['scaled_kp_depth'],
            est_depth_scale=est_depth_scale,
            est_depth_shift=est_depth_shift,
            target_shape_attr=self.target.shape_attr_values,
            target_depth=self.target.depth,
            target_kp_depth=self.target.keypoints_2d_depth,
            rendered_normals=rendered_normals,
            target_normals=target_normals,
            normal_mask=normal_mask,
            rendered_depth_map=rendered_depth_map,
            target_depth_map=target_depth_map,
            depth_map_mask=depth_map_mask,
        )
        total_loss.backward()
        # Two separate failure modes, both fatal for a long warm-started chain
        # (NaN params warm-start every later frame):
        # 1. A gradient spike blowing Adam's step up  -> clip the norm.
        # 2. A NON-FINITE gradient. The anny shape model's backward pass returns
        #    NaN on all six shape params at isolated exact values (e.g. weight=0.96,
        #    height=0.42 in a sweep of P111009's pose) -- a measure-zero interpolation
        #    singularity Adam/clipping hits in practice. clip_grad_norm_ on a NaN norm
        #    scales EVERY gradient to NaN and poisons all parameters, so zero the bad
        #    gradients first and nudge the affected shape params off the singular
        #    point (a 2e-3 jitter is far wider than the singular region).
        opt_params = [p for group in optimizer.param_groups for p in group['params']]
        for p in opt_params:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                if p.numel() == 1:  # scalar shape phenotype
                    p.data.add_(torch.empty_like(p).uniform_(-2e-3, 2e-3)).clamp_(0.0, 1.0)
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=5.0)
        optimizer.step()

        return loss_dict

    def _calibrate_depth_shift(self):
        """One-time init of depth_shift so the outlier-gated depth-map loss isn't
        zero (and gradient-less) everywhere on step 0. AnnyFIT's raw Z has an
        unconstrained offset from UniDepth's metric scale — depth_scale/depth_shift
        exist to learn it, but a hard delta-gate can't bootstrap from an arbitrary
        initial offset of order meters, so seed depth_shift from the current
        median residual instead of leaving it at 0.0."""
        with torch.no_grad():
            anny_output = self.anny_model.forward()
            coco_joints = anny_output['coco_joints']
            _, _, _, rendered_depth, target_depth, depth_mask = \
                self._render_scene(anny_output['vertices'], coco_joints,
                                   need_normal=False, need_depth=True)
            valid = depth_mask[0, 0] > 0
            if valid.sum() > 0:
                shift = (target_depth[0, 0][valid] - rendered_depth[0, 0][valid]).median()
                self.anny_model.depth_params['depth_shift'].data.fill_(shift.item())

    def optimize_stage(self, stage, stage_idx, global_step):
        """Optimizes the model for a single stage for the required number of epochs."""
        self.fitting_loss.update_weights(stage.loss_weights)

        if (self.fitting_loss.depth_map_weight > 0
                and self.target.depth_map.numel() > 0
                and not getattr(self, '_depth_shift_calibrated', False)):
            self._calibrate_depth_shift()
            self._depth_shift_calibrated = True

        optimizer = self.init_optimizer(stage)
        self.img_prefix = f"{self.img_name[:-4]}_{stage_idx}"

        if stage_idx == 0:
            self.initial_vertices = self.anny_model.verts_init.clone().detach()  # (bs, V, 3)

        for step_idx in tqdm(range(stage.epochs), desc=f"Stage {stage.name}"):
            losses = self._perform_one_optimization_step(optimizer, step_idx)
            self.log_step(losses, global_step, stage_idx, stage)
            global_step += 1

        # update the initial parameters for the next stage
        self.anny_model.update_init_parameters()

        # save final mesh
        save_path = os.path.join(self.vis_dir, f"{self.img_prefix}_final.jpg")
        self.visualize_mesh(vertices=self.anny_model.verts_init, save_path=save_path)

        return global_step


class TargetData(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('keypoints_2d', torch.tensor([]))
        self.register_buffer('dense_kp', torch.tensor([]))
        self.register_buffer('shape_attr_batch_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('shape_attr_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('shape_attr_values', torch.tensor([]))
        self.register_buffer('depth', torch.tensor([]))
        self.register_buffer('keypoints_2d_depth', torch.tensor([]))
        self.register_buffer('masks', torch.tensor([]))
        self.register_buffer('depth_map', torch.tensor([]))
        self.register_buffer('sapiens_normal', torch.tensor([]))

    def set_data(self, key: str, value: torch.Tensor):
        setattr(self, key, value)

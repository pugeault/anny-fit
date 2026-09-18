import numpy as np
import torch
import pickle
import roma
from dataclasses import dataclass
from collections import OrderedDict

import anny
from utils import clamp_but_preserve_gradients

# SMPL-X joint index → COCO-17 mapping (anny 163-joint convention)
_SMPLX_TO_COCO17 = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

class MyParameterDict(torch.nn.Module):
    """
    Dictionary of parameters that maps '.' into '_' to overcome PyTorch limitations.
    Useful for storing parameters where original keys include dots, such as joint names.
    """
    def __init__(self, params_tuple):
        super().__init__()
        self.key_mapping = OrderedDict()
        params = []
        self.param_names = []
        for (key, value) in params_tuple:
            valid_key = key.replace(".", "_")
            self.key_mapping[key] = valid_key
            params.append((valid_key, value))
            self.param_names.append(valid_key)
        assert len(set(self.key_mapping.values())) == len(self.key_mapping), "Collision during key mapping!"
        self.params = torch.nn.ParameterDict(params)

    def __getitem__(self, key):
        return self.params[self.key_mapping[key]]
    
    def keys(self):
        return self.key_mapping.keys()
    
    def param_keys(self):
        return self.param_names

    def values(self):
        return self.params.values()

    def items(self):
        for key in self.keys():
            yield key, self.__getitem__(key)

    def index(self, key):
        return self.param_names.index(self.key_mapping[key])

class Anny(torch.nn.Module):
    """ Extension of the Anny body model to support more joints and handle parameterization """

    def __init__(self, dtype=torch.float32, skinning_method='lbs', batch_size=1):
        super().__init__()
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = anny.create_fullbody_model(remove_unattached_vertices=False,
                                                local_changes=True,
                                                pose_parameterization='local-bone',
                                                topology='smplx',
                                                ).to(dtype=self.dtype)
        self.model.set_skinning_method(skinning_method)
        self.faces = self.model.get_triangular_faces()
        
        # joint limit ranges
        joint_limits = [
            (label, torch.nn.Parameter(
                torch.deg2rad(torch.stack([torch.as_tensor([-180., 180.]) for axis in 'xyz'], dim=-1)),
                requires_grad=False  # Tell PyTorch not to track gradients for these
            ))
            for label in self.model.bone_labels if label != "root"
        ]
        
        self.joint_limit_ranges = MyParameterDict(joint_limits)

        self.register_buffer('identity_rotation', torch.zeros(1, 3, dtype=dtype))
        self.register_buffer('null_translation', torch.zeros(1, 3, dtype=dtype))

        self.root_rotation_params = torch.nn.Parameter(torch.zeros(self.batch_size, 3, dtype=dtype).contiguous().requires_grad_(True))
        self.root_translation_params = torch.nn.Parameter(torch.zeros(self.batch_size, 1, 3, dtype=dtype, requires_grad=True))
        self.joints_rotation_params = MyParameterDict([(label, torch.zeros(self.batch_size, 3, dtype=dtype, requires_grad=True)) for label in self.joint_limit_ranges.keys()])
        # to maintain order of the parameters use list of tuples
        self.shape_params = torch.nn.ParameterDict([(key, torch.nn.Parameter(torch.full((self.batch_size,), fill_value=0.5, dtype=dtype, requires_grad=True))) for key in self.model.phenotype_labels])
        # not optimizing the local changes
        self.local_changes_kwargs = torch.nn.ParameterDict([(key, torch.nn.Parameter(torch.zeros(self.batch_size, dtype=dtype, requires_grad=False))) for key in self.model.local_change_labels]) 

        # Joint index buffer: anny 163-joint → COCO-17
        self.register_buffer('smplx_to_coco17', torch.tensor(_SMPLX_TO_COCO17, dtype=torch.long))

        self.face_params_names = self.get_face_parameter_names()
        self.fingertoe_params_names = self.get_fingertoe_parameter_names()

        # parameters for depth scaling
        self.depth_params = torch.nn.ParameterDict({
            'depth_scale': torch.nn.Parameter(torch.tensor(1.0, dtype=self.dtype), requires_grad=True),
            'depth_shift': torch.nn.Parameter(torch.tensor(0.0, dtype=self.dtype), requires_grad=True)
        })

        
    def update_parameters(self, root_rotation_params=None, root_translation_params=None, body_pose_params=None, shape_params=None):
        """
        Updates the parameters of the model.
        Args:
            root_rotation_params (torch.Tensor): Rotation parameters for the root joint.
            root_translation_params (torch.Tensor): Translation parameters for the root joint.
            body_pose_params (dict): Body pose parameters for each joint.
            shape_params (dict): Shape parameters for macrodetails.
        """
        if root_rotation_params is not None:
            self.root_rotation_params.data = root_rotation_params.to(dtype=self.dtype)
        if root_translation_params is not None:
            self.root_translation_params.data = root_translation_params.to(dtype=self.dtype)
        if body_pose_params is not None:
            for idx, label in enumerate(self.joints_rotation_params.keys()):
                self.joints_rotation_params[label].data = body_pose_params[:, idx].to(dtype=self.dtype)
        if shape_params is not None:
            for idx, key in enumerate(self.shape_params.keys()):
                self.shape_params[key].data = shape_params[:, idx].to(dtype=self.dtype)

    def init_parameters(self, init_parameters):
        """
        Initializes the parameters of the model.
        Args:
            init_parameters (dict): Dictionary containing initial parameters for the model.
        """

        self.update_parameters(
            root_rotation_params=init_parameters.get('root_rotation_params', None),
            root_translation_params=init_parameters.get('root_translation_params', None),
            body_pose_params=init_parameters.get('body_pose_params', None),
            shape_params=init_parameters.get('shape_params', None)
        )
        # save the initial parameters for the loss
        self.init_pose = torch.nn.Parameter(torch.stack(list(self.joints_rotation_params.values()), dim=1))  # (bs, 162, 3)
        self.init_shape = torch.nn.Parameter(torch.stack(list(self.shape_params.values()), dim=1))  # (bs, 8)
        with torch.no_grad():
            v3d = self.forward()['vertices']

        self.verts_init = torch.nn.Parameter(v3d)

    def update_init_parameters(self):
        """
        Use a clone of the current parameters to update the initial parameters.
        """
        self.init_pose.data = torch.stack(list(self.joints_rotation_params.values()), dim=1).clone().detach()
        self.init_shape.data = torch.stack(list(self.shape_params.values()), dim=1).clone().detach()
        with torch.no_grad():
            v3d = self.forward()['vertices']
        self.verts_init.data = v3d.clone().detach()

    def get_face_parameter_names(self):
        """
        Returns the names of the parameters used for the face.
        """
        face_prefixes = ["oris", "levator", "eye", "temporalis", "orbicularis", "oculi", "risorius", "jaw", "tongue", "special"]
        # find full names in rotation parameters
        face_params = [name for name in self.joints_rotation_params.keys() if any(name.startswith(prefix) for prefix in face_prefixes)]
        return face_params
    
    def get_fingertoe_parameter_names(self):
        """
        Returns the names of the parameters used for the fingers and toes.
        """
        fingertoe_prefixes = ["toe", "finger", "metacarpal"]
        # find full names in rotation parameters
        fingertoe_params = [name for name in self.joints_rotation_params.keys() if any(name.startswith(prefix) for prefix in fingertoe_prefixes)]
        return fingertoe_params
    
    def get_bodypose_parameters(self):
        """
        Returns body pose parameters as axis-angle rotation vectors.
        """
        keys = list(self.joints_rotation_params.keys())
        values = torch.stack(list(self.joints_rotation_params.values()), dim=1)  # (bs, 162, 3)
        return keys, values
    
    def get_shape_parameters(self):
        """
        Returns shape parameters as a vector.
        """
        keys = list(self.shape_params.keys())
        values = torch.stack(list(self.shape_params.values()), dim=1)  # (bs, num_phenotypes)
        return keys, values
    
    def get_local_changes_parameters(self):
        """
        Returns local changes parameters as a vector.
        """
        keys = list(self.local_changes_kwargs.keys())
        values = torch.stack(list(self.local_changes_kwargs.values()), dim=1)
        return keys, values

    def get_parametrization(self):
        """
        Computes the current parameterization of the model:
        - Rigid transformations (as `roma.Rigid`) for each joint
        - Macrodetails (e.g., age, muscle tone, race blend)
        - Local shape refinements
        Returns:
            pose_parameters (dict): mapping from joint name to roma.Rigid
            phenotype_kwargs (dict): macro shape parameters
            local_changes_kwargs (dict): fine-level deformations
        """
        pose_parameters = dict()
        root_rotmat = roma.rotvec_to_rotmat(self.root_rotation_params)
        root_transl = self.root_translation_params.view(self.batch_size, 3)
        pose_parameters["root"] = roma.Rigid(roma.special_procrustes(root_rotmat, regularization=0.1), root_transl)

        # Vectorized: clamp + rotvec_to_rotmat for all 162 non-root bones in ONE
        # batched call instead of 162 separate small ops. Every bone's
        # joint_limit_ranges is identical ([-pi,pi] per axis, see __init__), so
        # the per-bone lookup was redundant work — this was ~140ms of a ~215ms
        # optimisation step (mostly CUDA kernel-launch overhead from the loop),
        # see annyfit_loss_tuning memory. Validated bit-for-bit identical
        # vertices/bone_poses/gradients against the old per-bone loop, 3x
        # faster on fwd+bwd. Falls back to the loop if the identical-ranges
        # assumption ever stops holding, so correctness can't silently drift.
        labels = list(self.joints_rotation_params.keys())
        ranges0 = self.joint_limit_ranges[labels[0]]  # (2, 3)
        ranges_uniform = all(torch.equal(self.joint_limit_ranges[l], ranges0) for l in labels)
        if ranges_uniform:
            stacked_params = torch.stack([self.joints_rotation_params[l] for l in labels], dim=1)  # (bs,162,3)
            clamped = clamp_but_preserve_gradients(stacked_params, ranges0[0, None, None], ranges0[1, None, None])
            rotmats = roma.rotvec_to_rotmat(clamped)  # (bs,162,3,3)
            null_transl = self.null_translation.view(1, 1, 3).expand(self.batch_size, len(labels), -1)
            for i, label in enumerate(labels):
                pose_parameters[label] = roma.Rigid(rotmats[:, i], null_transl[:, i])
        else:
            for label, param in self.joints_rotation_params.items():
                ranges = self.joint_limit_ranges[label]
                clamped_param = clamp_but_preserve_gradients(param, ranges[0,None], ranges[1,None])
                rotmat = roma.rotvec_to_rotmat(clamped_param)
                pose_parameters[label] = roma.Rigid(rotmat, self.null_translation.expand(self.batch_size, -1))

        shape_kwargs = dict()
        for key, value in self.shape_params.items():
            shape_kwargs[key] = clamp_but_preserve_gradients(value, 0, 1)

        phenotype_kwargs = dict()
        phenotype_kwargs.update(shape_kwargs)
        
        return pose_parameters, phenotype_kwargs, self.local_changes_kwargs
    
    def get_joints(self, bone_poses):
        """bone_poses (B,163,4,4) → all 163 joints (B,163,3), zeros for dense."""
        j3d = bone_poses[:, :, :3, 3]                                    # (B, 163, 3)
        dense_kps = torch.zeros(j3d.shape[0], 138, 3,
                                device=j3d.device, dtype=j3d.dtype)      # (B, 138, 3) — disabled
        return j3d, dense_kps

    def forward(self):
        """
        Performs forward thought the body model.
        """

        # convert to rotmat representation
        pose_parameters, phenotype_kwargs, local_changes_kwargs = self.get_parametrization()

        # run the model
        output = self.model(pose_parameters=pose_parameters,
                            phenotype_kwargs=phenotype_kwargs,
                            local_changes_kwargs=local_changes_kwargs)

        v3d = output['vertices']
        bone_poses = output['bone_poses']              # (B, 163, 4, 4)

        coco_joints, dense_joints = self.get_joints(bone_poses)  # 17 COCO + 138 dense
        
        shape = torch.stack(list(self.shape_params.values()), dim=1) # (bs, 11)
        body_pose = torch.stack(list(self.joints_rotation_params.values()), dim=1) # (bs, 162, 3)

        # scale root depth
        root_transl_flat = self.root_translation_params.view(self.batch_size, 3)
        scaled_depth = root_transl_flat[:, 2] * self.depth_params['depth_scale'] + self.depth_params['depth_shift']
        scaled_kp_depth = coco_joints[:, :, 2] * self.depth_params['depth_scale'] + self.depth_params['depth_shift']

        # print(f"depth: {self.root_translation_params[:, 2]}, scaled: {scaled_depth}, scale: {self.depth_params['depth_scale'].item()}, shift: {self.depth_params['depth_shift'].item()}")

        final_output = {'vertices': v3d,
                        'smpl_vertices': v3d,  # keep key for compatibility; no SMPL regressor
                        'coco_joints': coco_joints, # coco 17 joints
                        'dense_joints': dense_joints,
                        'shape': shape,
                        'shape_dict': self.shape_params,
                        'body_pose': body_pose,
                        'root_rotation': self.root_rotation_params,
                        'root_translation': self.root_translation_params,
                        'scaled_depth': scaled_depth,
                        'scaled_kp_depth': scaled_kp_depth,
                        'depth_scale': self.depth_params['depth_scale'],
                        'depth_shift': self.depth_params['depth_shift'],
                        }

        return final_output
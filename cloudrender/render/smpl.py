import numpy as np
import torch
from torch.nn import Module
import smplx
from loguru import logger
from typing import Dict

from .mesh import TexturedMesh, SimpleMesh, Mesh
from .renderable import DynamicTimedRenderable
from .utils import MeshNorms


class SMPLXModelBase(DynamicTimedRenderable):
    MODEL_PARAM_NAMES = {
        "smpl": ["betas", "body_pose", "global_orient", "transl"],
        "smplh": ["betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose"],
        "smplx": ["betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose", "expression", "jaw_pose", "leye_pose",
                  "reye_pose"],
    }

    def __init__(self, device=None, smpl_root=None, template=None, gender="neutral", flat_hand_mean=True, model_type="smpl", global_offset=None,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = None
        self.smpl_root = smpl_root
        self.device = torch.device(device if device is not None else "cpu")
        self.template = template
        self.global_offset = global_offset
        self.model_type = model_type
        smpl_compatible = False
        if self.smpl_root is None:
            self.smpl_root = "./models"
        if "compat" in self.model_type:
            self.model_type = self.model_type.split("_")[0]
            smpl_compatible = True
        self.available_params = SMPLXModelBase.MODEL_PARAM_NAMES[self.model_type]
        self._init_model(gender, smpl_compatible, flat_hand_mean)
        self.nglverts = len(self.get_vertices()[0])

    def _init_model(self, gender='neutral', smpl_compatible=False, flat_hand_mean=True):
        self.model_layer = smplx.create(self.smpl_root, model_type=self.model_type, gender=gender, use_pca=False, flat_hand_mean=flat_hand_mean).to(
            self.device)
        self.model_layer.requires_grad_(False)
        if smpl_compatible:
            smpl_model = smplx.create(self.smpl_root, model_type="smpl", gender=gender)
            self.model_layer.shapedirs[:] = smpl_model.shapedirs.detach().to(self.device)
        if self.template is not None:
            self.model_layer.v_template[:] = torch.tensor(self.template, dtype=self.model_layer.v_template.dtype,
                                                          device=self.device)
        if self.global_offset is not None:
            self.model_layer.v_template[:] += torch.tensor(self.global_offset[np.newaxis, :], dtype=self.model_layer.v_template.dtype,
                                                           device=self.device)
        self.normals_layer = MeshNorms(
            self.model_layer.faces_tensor)  # torch.tensor(self.model_layer.faces.astype(np.long), dtype=torch.long, device=self.device))
        self.gender = gender
        self.smpl_compatible = smpl_compatible
        self._current_params = {x: getattr(self.model_layer, x).squeeze(0).clone() for x in self.available_params}

    def _preprocess_param(self, param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float32)
        param = param.to(self.device)
        return param

    def _finalize_init(self):
        self.faces_numpy = self.model_layer.faces.astype(np.long)
        self.faces = self.model_layer.faces_tensor  # torch.tensor(self.model_layer.faces.astype(np.long), dtype=torch.long, device=self.device)
        self.flat_faces = self.faces.view(-1)

    def update_params(self, **model_params):
        for param_name, param_val in model_params.items():
            if param_name in self.available_params:
                param_val = self._preprocess_param(param_val)
                self._current_params[param_name] = param_val

    def get_vertices(self, return_normals=True, **model_params):
        self.update_params(**model_params)
        batch_params = {x: self._current_params[x].unsqueeze(0) for x in self.available_params}
        output = self.model_layer(**batch_params)
        verts = output.vertices.squeeze(0)
        if return_normals:
            normals = self.normals_layer.vertices_norms(verts)
            return verts, normals
        else:
            return verts

    def get_mesh(self, **model_params):
        verts, normals = self.get_vertices(**model_params)
        mesh = Mesh.MeshContainer(verts.cpu().numpy(), self.faces_numpy, vertex_normals=normals.cpu().numpy())
        return mesh

    def get_joints(self, **model_params):
        self.update_params(**model_params)
        batch_params = {x: self._current_params[x].unsqueeze(0) for x in self.available_params}
        output = self.model_layer(**batch_params)
        joints = output.joints.squeeze(0)
        return joints.cpu().numpy()

    def _set_sequence(self, params_seq):
        self.params_sequence = params_seq
        self.sequence_len = len(params_seq)

    def _load_current_frame(self):
        params = self.params_sequence[self.current_sequence_frame_ind]
        self.update_buffers(**params)

    @property
    def global_translation(self) -> np.ndarray:
        return self._current_params["transl"].cpu().numpy()

    @property
    def current_params(self) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self._current_params.items()}


class SMPLXColoredModel(SimpleMesh, SMPLXModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_uniform_color()

    def set_uniform_color(self, color=(200, 200, 200, 255)):
        self.color = color
        self.vertex_colors = np.tile(np.array(color, dtype=np.uint8).reshape(1, 4), (self.nglverts, 1))

    def _set_buffers(self, **model_params):
        mesh = self.get_mesh(**model_params)
        mesh.colors = self.vertex_colors
        super()._set_buffers(mesh)

    def _update_buffers(self, **model_params):
        mesh = self.get_mesh(**model_params)
        mesh.colors = self.vertex_colors
        super()._update_buffers(mesh)


class SMPLXTexturedModel(TexturedMesh, SMPLXModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_texture = False

    def _set_buffers(self, **model_params):
        mesh = self.get_mesh(**model_params)
        if self.update_texture:
            mesh.texture = self.texture
            mesh.face_uv_map = self.uv_map
            self.update_texture = False
        super()._set_buffers(mesh)

    def _update_buffers(self, **model_params):
        mesh = self.get_mesh(**model_params)
        if self.update_texture:
            mesh.texture = self.texture
            mesh.face_uv_map = self.uv_map
            self.update_texture = False
        super()._update_buffers(mesh)

    def set_texture(self, texture, uv_map):
        self.texture = texture
        self.uv_map = uv_map
        self.update_texture = True

    def set_uniform_color(self, color=(200, 200, 200, 255)):
        pass

import numpy as np
import torch
import smplx

from .mesh import SimpleMesh
from .renderable import DynamicTimedRenderable
from .utils import MeshNorms


class SMPLModel(SimpleMesh):
    def __init__(self, device=None, smpl_root=None, template=None, gender="neutral", model_type="smpl", global_offset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = None
        self.smpl_root = smpl_root
        self.device = torch.device(device if device is not None else "cpu")
        self.pose_params = torch.zeros(72, device=self.device)
        self.translation_params = torch.zeros(3, device=self.device)
        self.template = template
        self.global_offset = global_offset
        self.model_type = model_type
        self.smpl_compatible = False
        if self.smpl_root is None:
            self.smpl_root = "./models"
        if "compat" in self.model_type:
            self.model_type = self.model_type.split("_")[0]
            self.smpl_compatible = True
        self._set_smpl(gender)
        self.nglverts = len(self.get_smpl()[0])
        self.set_uniform_color()

    def _set_smpl(self, gender='neutral', shape_params=None):
        self.model_layer = smplx.create(self.smpl_root, model_type=self.model_type, gender=gender).to(self.device)
        self.model_layer.requires_grad_(False)
        if self.smpl_compatible:
            smpl_model = smplx.create(self.smpl_root, model_type="smpl", gender=gender)
            self.model_layer.shapedirs[:] = smpl_model.shapedirs.detach().to(self.device)
        if self.template is not None:
            self.model_layer.v_template[:] = torch.tensor(self.template, dtype=self.model_layer.v_template.dtype,
                                                          device=self.device)
        if self.global_offset is not None:
            self.model_layer.v_template[:] += torch.tensor(self.global_offset[np.newaxis, :], dtype=self.model_layer.v_template.dtype,
                                                           device=self.device)
        self.normals_layer = MeshNorms(self.model_layer.faces_tensor) #torch.tensor(self.model_layer.faces.astype(int), dtype=torch.long, device=self.device))
        self.gender = gender
        self.shape_params = torch.zeros(10, device=self.device) if shape_params is None else \
            torch.tensor(shape_params, dtype=torch.float32, device=self.device)

    def _preprocess_param(self, param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float32)
        param = param.to(self.device)
        return param

    def _finalize_init(self):
        super()._finalize_init()
        self.faces_numpy = self.model_layer.faces.astype(int)
        self.faces = self.model_layer.faces_tensor #torch.tensor(self.model_layer.faces.astype(int), dtype=torch.long, device=self.device)
        self.flat_faces = self.faces.view(-1)

    def update_params(self, pose=None, shape=None, translation=None):
        if pose is not None:
            self.pose_params = self._preprocess_param(pose)
        if shape is not None:
            self.shape_params = self._preprocess_param(shape)
        if translation is not None:
            self.translation_params = self._preprocess_param(translation)

    def set_uniform_color(self, color=(200, 200, 200, 255)):
        self.color = color
        self.vertex_colors = np.tile(np.array(color, dtype=np.uint8).reshape(1, 4), (self.nglverts, 1))

    def get_smpl(self, pose_params=None, shape_params=None, translation_params=None):
        self.update_params(pose_params, shape_params, translation_params)
        batch_pose_params = self.pose_params.unsqueeze(0)
        batch_shape_params = self.shape_params.unsqueeze(0)
        if self.model_type == "smplh":
            batch_pose_params = batch_pose_params[:,:-6]
        output = self.model_layer(global_orient =batch_pose_params[:, :3],
                                  body_pose=batch_pose_params[:,3:], betas=batch_shape_params)
        verts = output.vertices
        normals = self.normals_layer.vertices_norms(verts.squeeze(0))
        return verts.squeeze(0) + self.translation_params.unsqueeze(0), normals

    def get_smpl_mesh(self, pose_params=None, shape_params=None, translation_params=None):
        verts, normals = self.get_smpl(pose_params, shape_params, translation_params)
        mesh = self.MeshContainer(verts.cpu().numpy(), self.faces_numpy, self.vertex_colors, normals.cpu().numpy())
        return mesh

    def _set_buffers(self, pose_params=None, shape_params=None, translation_params=None):
        mesh = self.get_smpl_mesh(pose_params, shape_params, translation_params)
        super()._set_buffers(mesh)

    def _update_buffers(self, pose_params=None, shape_params=None, translation_params=None):
        mesh = self.get_smpl_mesh(pose_params, shape_params, translation_params)
        super()._update_buffers(mesh)


class AnimatableSMPLModel(SMPLModel, DynamicTimedRenderable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_sequence(self, params_seq):
        self.params_sequence = params_seq
        self.sequence_len = len(params_seq)

    def _load_current_frame(self):
        params = self.params_sequence[self.current_sequence_frame_ind]
        pose = params['pose'] if 'pose' in params else None
        shape = params['shape'] if 'shape' in params else None
        translation = params['translation'] if 'translation' in params else None
        self.update_buffers(pose, shape, translation)

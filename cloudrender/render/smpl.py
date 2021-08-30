import numpy as np
import torch
from torch.nn import Module
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

from .mesh import SimpleMesh
from .renderable import DynamicTimedRenderable

class MeshNorms(Module):
    def __init__(self, faces):
        super().__init__()
        self.faces_count = faces.size(0)
        normmap = self.compute_face2verts_normmap(faces)
        self.register_buffer("normmap", normmap)
        self.register_buffer("faces", faces)

    @staticmethod
    def compute_face2verts_normmap(faces):
        _, faces_in_vertices_count = torch.unique(faces, sorted=True, return_counts=True)
        verts_count = len(faces_in_vertices_count)
        faces_in_vertex_max = faces_in_vertices_count.max().item()
        faces_appearance = torch.argsort(faces.view(-1))//3
        appearance_array_off = 0
        # print(faces.size())
        normmap = torch.ones(verts_count, faces_in_vertex_max, dtype=torch.long, device=faces.device)*faces.size(0)
        for i in range(verts_count):
            faces_in_vertex = faces_in_vertices_count[i]
            normmap[i, :faces_in_vertex] = faces_appearance[appearance_array_off:appearance_array_off + faces_in_vertex]
            appearance_array_off += faces_in_vertex
        return normmap

    def faces_norms(self, verts):
        verts_size = verts.size()
        verts_faces = verts[..., self.faces.view(-1), :].view(*verts_size[:-2],-1,3,3)
        vct1 = verts_faces[..., 0, :] - verts_faces[..., 1, :]
        vct2 = verts_faces[..., 0, :] - verts_faces[..., 2, :]
        cross = torch.cross(vct1, vct2, dim=-1)
        faces_norms = cross/torch.norm(cross, dim=-1, keepdim=True)
        return faces_norms

    def vertices_norms(self, verts):
        faces_norms = self.faces_norms(verts)
        faces_norms = torch.cat([faces_norms, torch.zeros(*faces_norms.size()[:-2], 1, 3, device=verts.device)], dim=-2)
        vertices_norms = faces_norms[..., self.normmap.view(-1), :].view(*faces_norms.size()[:-2],-1,
                                                                         self.normmap.size(1), 3).sum(dim=-2)
        vertices_norms = vertices_norms/torch.norm(vertices_norms, dim=-1, keepdim=True)
        return vertices_norms


class SMPLModel(SimpleMesh):
    def __init__(self, device=None, smpl_root=None, gender="neutral", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = None
        self.smpl_root = smpl_root
        self.device = torch.device(device if device is not None else "cpu")
        self.pose_params = torch.zeros(72, device=self.device)
        self.translation_params = torch.zeros(3, device=self.device)
        self._set_smpl(gender)
        self.nglverts = len(self.get_smpl()[0])
        self.set_uniform_color()

    def _set_smpl(self, gender='neutral', shape_params=None):
        if self.smpl_root is None:
            self.smpl_layer = SMPL_Layer(center_idx=0, gender=gender).to(self.device)
        else:
            self.smpl_layer = SMPL_Layer(center_idx=0, gender=gender, model_root=self.smpl_root).to(self.device)
        self.normals_layer = MeshNorms(self.smpl_layer.th_faces)
        self.gender = gender
        self.shape_params = torch.zeros(10, device=self.device) if shape_params is None else \
            torch.tensor(shape_params, dtype=torch.float32, device=self.device)

    def _preprocess_param(self, param):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype = torch.float32)
        param = param.to(self.device)
        return param

    def _finalize_init(self):
        super()._finalize_init()
        self.faces = self.smpl_layer.th_faces
        self.faces_numpy = self.faces.cpu().numpy()
        self.flat_faces = self.smpl_layer.th_faces.view(-1)

    def update_params(self, pose = None, shape = None, translation = None):
        if pose is not None:
            self.pose_params = self._preprocess_param(pose)
        if shape is not None:
            self.shape_params = self._preprocess_param(shape)
        if translation is not None:
            self.translation_params = self._preprocess_param(translation)

    def set_uniform_color(self, color=(200, 200, 200, 255)):
        self.color = color
        self.vertex_colors = np.tile(np.array(color, dtype=np.uint8).reshape(1,4), (self.nglverts, 1))

    def get_smpl(self, pose_params = None, shape_params = None, translation_params = None):
        self.update_params(pose_params, shape_params, translation_params)
        verts, joints = self.smpl_layer(th_pose_axisang=self.pose_params.unsqueeze(0),
                                                 th_betas=self.shape_params.unsqueeze(0))
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




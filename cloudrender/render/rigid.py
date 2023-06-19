import numpy as np
import torch
from torch.nn import Module
import smplx
from loguru import logger
from typing import Dict, List

from .mesh import TexturedMesh, SimpleMesh, Mesh
from .pointcloud import SimplePointcloud
from .renderable import DynamicTimedRenderable
from .utils import MeshNorms, centrify_smplx_root_joint
from ..utils import get_closest_ind_after, get_closest_ind_before, ObjectTrajectory



class RigidObject(DynamicTimedRenderable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_sequence(self, params_seq: ObjectTrajectory):
        self.params_sequence = params_seq
        self.sequence_len = len(params_seq)

    def _load_current_frame(self):
        obj_location = self.params_sequence[self.current_sequence_frame_ind]
        obj_model_position = obj_location["position"]
        obj_model_quat = obj_location["quaternion"]
        self.init_model_extrinsics(obj_model_quat, obj_model_position)

class RigidObjectSimpleMesh(SimpleMesh, RigidObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class RigidObjectSimplePointcloud(SimplePointcloud, RigidObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



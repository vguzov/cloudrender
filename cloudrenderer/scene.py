import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Sequence
from .render.renderable import Renderable
from .camera import BaseCameraModel, OrthogonalCameraModel
from .render.lights import Light, DirectionalLight
from .render.shadowmap import ShadowMap


class Scene:
    def __init__(self):
        self.objects: List[Renderable] = []
        self.lights: List[Light] = []
        self.shadowmaps: List[ShadowMap] = []
        self.camera: BaseCameraModel = None

    def set_camera(self, camera: BaseCameraModel):
        self.camera = camera
        for renderable_object in self.objects:
            renderable_object.set_camera(camera)

    def add_object(self, obj: Renderable):
        self.objects.append(obj)

    def add_light(self, light: Light):
        self.lights.append(light)

    def add_shadowmap(self, shadowmap: ShadowMap):
        self.shadowmaps.append(shadowmap)

    def add_dirlight_with_shadow(self, light: DirectionalLight, shadowmap_texsize: Sequence[int],
                                 shadowmap_worldsize: Sequence[float], shadowmap_center: Sequence[float]):
        self.add_light(light)
        light_camera = OrthogonalCameraModel()
        shadowmap_worldsize = np.array(shadowmap_worldsize)
        shadowmap_center = np.array(shadowmap_center)
        shadowmap_halfsize = shadowmap_worldsize/2
        near = 0
        mincorner = np.array([-shadowmap_halfsize[0], -shadowmap_halfsize[1], near-shadowmap_halfsize[2]])
        maxcorner = np.array([shadowmap_halfsize[0], shadowmap_halfsize[1], near+shadowmap_halfsize[2]])
        light_camera.init_intrinsics(shadowmap_texsize,mincorner[0], maxcorner[0],
                                     mincorner[1], maxcorner[1], mincorner[2], maxcorner[2])
        current_dir = np.array([0,0,-1.])
        target_dir = light.direction
        min_rot_vector = np.cross(current_dir, target_dir)
        quat = np.roll(Rotation.from_rotvec(min_rot_vector).as_quat(), 1)
        light_camera.init_extrinsics(quat, shadowmap_center)
        shadowmap = ShadowMap(light_camera, shadowmap_texsize)
        self.add_shadowmap(shadowmap)
        return shadowmap

    def draw(self, reset=True):
        shading_objects = [o for o in self.objects if o.generate_shadows]
        for shadowmap in self.shadowmaps:
            shadowmap.update_shadowmap(shading_objects)

        is_buffer_changed = False
        for renderable_object in self.objects:
            if renderable_object.draw_shadows:
                is_buffer_changed |= renderable_object.draw(reset, self.lights, self.shadowmaps)
            else:
                is_buffer_changed |= renderable_object.draw(reset, self.lights)
        return is_buffer_changed

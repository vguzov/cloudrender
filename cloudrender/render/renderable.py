import numpy as np
import glm
from typing import List, Dict
from collections import defaultdict
from scipy.spatial.transform import Rotation
from OpenGL import GL as gl
from ..camera.models import BaseCameraModel, StandardProjectionCameraModel
from .lights import Light
from .shadowmap import ShadowMap
from .shaders.shader_loader import Shader


class Renderable:
    SHADOWMAPS_MAX = 6
    LIGHTS_MAX = {"directional": 1}

    class GLContext(object):
        pass

    @staticmethod
    def locate_uniforms(shader: Shader, keys: List[str]) -> Dict[str, int]:
        """
        Locates uniforms inside the supplied shader
        Args:
            shader (Shader): shader to seek uniforms in
            keys: list of uniforms names
        Returns:
            Dict[str, int]: dict with uniforms locations
        """
        shader_ids = {k: gl.glGetUniformLocation(shader.program, k) for k in keys}
        return shader_ids

    def __init__(self, camera: BaseCameraModel = None, draw_shadows: bool = False, generate_shadows: bool = False):
        self.camera = camera
        # whether to draw this object during the next draw() pass
        self.visible = True
        # defines overall initialization progress
        self.initialized = False
        # defines whether OpenGL buffers and shaders are loaded
        self.context_initialized = False
        # additional features what requires shader replacement
        self.shader_mode = ''
        # defines whether object supports drawing shadow during rendering
        self.draw_shadows = draw_shadows
        # defines whether object supports shadowmap update
        self.generate_shadows = generate_shadows
        # defines whether several draw() calls are needed to fully draw the object
        self.is_progressive = False
        # defines color of the shadow
        self.shadowcolor = np.array([0, 0, 0, 0.5])

    def check_lights(self, lights: List[Light]):
        lights_count = defaultdict(int)
        for light in lights:
            lights_count[light.type] += 1
        for light_type, count in lights_count.items():
            if light_type not in self.LIGHTS_MAX:
                raise NotImplementedError(f"Light '{light_type}' is not supported for this object")
            if self.LIGHTS_MAX[light_type] < count:
                raise NotImplementedError(f"No more than {self.LIGHTS_MAX[light_type]} light of type '{light_type}'"
                                          f"are supported, got {count}")

    def check_shadowmaps(self, shadowmaps):
        assert len(shadowmaps) == 0 or self.draw_shadows, "Shadow drawing is disabled for that object"
        assert len(shadowmaps) < self.SHADOWMAPS_MAX, f"No more than {self.SHADOWMAPS_MAX} are supported"

    def draw(self, reset: bool = True, lights: List[Light] = None, shadowmaps: List[ShadowMap] = None) -> bool:
        """
        Main draw pass
        Args:
            reset (bool): Reset drawing progress (for progressive drawing)
            lights (List[Light]): All light objects that insfuence the current object
            shadowmaps (List[ShadowMap]): List of shadowmaps to draw shadows from
        Returns:
            bool: if drawing buffer was changed (if something was actually drawn)
        """
        if not self.visible:
            return False
        lights = [] if lights is None else lights
        shadowmaps = [] if shadowmaps is None else shadowmaps
        self.check_lights(lights)
        self.check_shadowmaps(shadowmaps)
        return self._draw(reset, lights, shadowmaps)

    def _draw(self, reset, lights, shadowmaps) -> bool:
        """
        Internal draw pass
        Args:
            reset (bool): Reset drawing progress (for progressive drawing)
            lights (List[Light]): All light objects that insfuence the current object
            shadowmaps (List[ShadowMap]): List of shadowmaps to draw shadows from
        Returns:
            bool: if drawing buffer was changed (if something was actually drawn)
        """
        return False

    def set_buffers(self, *args, **kwargs):
        """
        Sets the content and prepares render buffers
        """
        if self.context_initialized:
            if self.initialized:
                self._delete_buffers()
            self._set_buffers(*args, **kwargs)
            self.initialized = True

    def update_buffers(self, *args, **kwargs):
        """
        Updates the content and render buffers
        """
        if not self.initialized:
            self.set_buffers(*args, **kwargs)
        else:
            self._update_buffers(*args, **kwargs)

    def delete_buffers(self):
        """
        Deletes the content and render buffers
        """
        if self.initialized:
            self._delete_buffers()
            self.initialized = False

    def _set_buffers(self, *args, **kwargs):
        pass

    def _update_buffers(self, *args, **kwargs):
        self.set_buffers(*args, **kwargs)

    def _delete_buffers(self):
        pass

    def _finalize_init(self):
        pass

    def _init_shaders(self, camera_model, shader_mode):
        self.shader = None
        self.shadowgen_shader = None

    def _reload_shaders(self, shader_mode: str = None):
        self._init_shaders(self.camera.model, shader_mode if shader_mode is not None else self.shader_mode)
        self.context.shader_ids.update(self.camera.locate_uniforms(self.shader))
        self.context.shader_ids.update(self.locate_uniforms(self.shader, ["M"]))
        if self.generate_shadows:
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader,
                                                                          StandardProjectionCameraModel.uniforms_names))
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader, ["M"]))


    def init_context(self, shader_mode: str = ''):
        """
        Inits some OpenGL buffers and loads shaders
        Args:
            shader_mode (str): additional features what requires shader replacement
        """
        assert self.camera is not None, "Camera must be set before context initialization"
        self.shader_mode = shader_mode
        self.context = self.GLContext()
        self.context.shader_ids = {}
        if self.generate_shadows:
            self.shadowgen_context = self.GLContext()
            self.shadowgen_context.shader_ids = {}
        self._reload_shaders()
        self.init_model_extrinsics(np.array([1.,0,0,0]), np.zeros(3))
        self._finalize_init()
        self.context_initialized = True

    def init_model_extrinsics(self, quat: np.ndarray, pose: np.ndarray):
        """
        Positions the object in the scene
        Args:
            quat: quaternion in WXYZ format stored in np array of shape (4,)
            pose: translation offset vector of shape (3,)
        """
        self.model_quat = quat
        self.model_pose = pose
        # Only cam/local2world supported here
        R = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        t = np.array([pose]).T
        RT = np.vstack([np.hstack([R, t]), [[0, 0, 0, 1]]])
        self.context.Model = glm.mat4(*(RT.T.astype(np.float32).copy()))

    def set_camera(self, camera: BaseCameraModel):
        """
        Sets the main rendering camera
        Args:
            camera (BaseCameraModel): the rendering camera
        """
        self.camera = camera
        if self.context_initialized:
            self._reload_shaders()

    def upload_uniforms(self, shader_ids: Dict[str, int], lights: List[Light], shadowmaps: List[ShadowMap]):
        """
        Upload all uniform variables for the main drawing pass
        Args:
            shader_ids: dictionary containing uniforms locations
            lights: list of lights affecting current draw pass
            shadowmaps: List of shadowmaps affecting current draw pass
        """
        self.camera.upload(shader_ids)
        gl.glUniformMatrix4fv(shader_ids['M'], 1, gl.GL_FALSE, glm.value_ptr(self.context.Model))
        self._upload_uniforms(shader_ids, lights, shadowmaps)

    def _upload_uniforms(self, shader_ids: Dict[str, int], lights: List[Light] = (), shadowmaps = ()):
        pass

    def upload_shadowgen_uniforms(self, shadowmap_camera: StandardProjectionCameraModel, shader_ids: dict):
        """
        Upload all uniform variables for the shadowmap update drawing pass
        Args:
           shadowmap_camera: perspective camera for shadow calculation
           shader_ids: dictionary containing uniforms locations
        """
        shadowmap_camera.upload(shader_ids)
        gl.glUniformMatrix4fv(shader_ids['M'], 1, gl.GL_FALSE, glm.value_ptr(self.context.Model))
        self._upload_shadowngen_uniforms(shader_ids)

    def _upload_shadowngen_uniforms(self, shader_ids):
        pass

    def draw_shadowmap(self, shadowmap_camera: StandardProjectionCameraModel):
        self._draw_shadowmap(shadowmap_camera)

    def _draw_shadowmap(self, shadowmap_camera: StandardProjectionCameraModel):
        pass


class DynamicRenderable(Renderable):
    def __init__(self, camera: BaseCameraModel = None, draw_shadows: bool = False, generate_shadows: bool = False,
                 *args, **kwargs):
        super().__init__(camera, draw_shadows, generate_shadows)
        self.sequence_initialized = False
        self.current_sequence_frame_ind = 0
        self.sequence_len = 0
        self.loaded_frame_ind = -1

    def set_sequence(self, *args, **kwargs):
        self._set_sequence(*args, **kwargs)
        assert self.sequence_len > 0, "Sequence length must be positive, make sure to set it during _set_sequence()"
        self.sequence_initialized = True

    def _set_sequence(self, *args, **kwargs):
        pass

    def unset_sequence(self):
        self._unset_sequence()
        self.sequence_initialized = False
        self.current_sequence_frame_ind = 0
        self.sequence_len = 0
        self.loaded_frame_ind = -1

    def _unset_sequence(self):
        pass

    def load_current_frame(self):
        if self.loaded_frame_ind != self.current_sequence_frame_ind:
            self._load_current_frame()
        self.loaded_frame_ind = self.current_sequence_frame_ind

    def reload_current_frame(self):
        self._load_current_frame()
        self.loaded_frame_ind = self.current_sequence_frame_ind

    def _load_current_frame(self):
        pass

    def reset_current_frame(self):
        self.current_sequence_frame_ind = 0
        self.load_current_frame()

    def set_current_frame(self, frame_index: int) -> bool:
        """
        Set the current frame for the next draw cycle
        Args:
            frame_index: frame index
        Returns:
            bool: whether the index was set successfully
        """
        if self.sequence_initialized and frame_index>=0 and frame_index<self.sequence_len:
            self.current_sequence_frame_ind = frame_index
            self.load_current_frame()
            return True
        else:
            return False

    def next_frame(self):
        return self.set_current_frame(self.current_sequence_frame_ind + 1)

    def prev_frame(self):
        return self.set_current_frame(self.current_sequence_frame_ind - 1)


class DynamicTimedRenderable(DynamicRenderable):
    def __init__(self, camera: BaseCameraModel = None, draw_shadows: bool = False, generate_shadows: bool = False,
                 *args, **kwargs):
        super().__init__(camera, draw_shadows, generate_shadows)
        self.time_offset = 0
        self.default_frame_time = 1/60.
        self.sequence_frame_times = None
        self.current_time = 0

    def set_sequence(self, *args, times = None, default_frame_time: float = 1./60, **kwargs):
        super().set_sequence(*args, **kwargs)
        self.default_frame_time = default_frame_time
        if times is not None:
            self.sequence_frame_times = np.array(times, dtype=np.float64)
        else:
            self.sequence_frame_times = np.arange(self.sequence_len, dtype=np.float64)*self.default_frame_time
        self.current_time = self.time_offset

    def unset_sequence(self):
        super().unset_sequence()
        self.sequence_frame_times = None
        self.current_time = self.time_offset

    def load_timed(self):
        if self.sequence_len > 0 and self.sequence_frame_times is not None:
            times_diff = (self.current_time - self.sequence_frame_times)
            mask = times_diff >= 0
            if mask.sum() == 0:
                index = 0
            else:
                masked_argmin = np.argmin(times_diff[mask])
                index = np.arange(times_diff.shape[0])[mask][masked_argmin]
            self.set_current_frame(index)

    def set_time(self, time):
        self.current_time = time+self.time_offset
        self.load_timed()

    def set_time_offset(self, offset):
        time_diff = offset-self.time_offset
        self.time_offset = offset
        self.advance_time(time_diff)

    def advance_time(self, time_delta):
        self.current_time += time_delta
        self.load_timed()

    def reset_time(self):
        self.current_time = self.time_offset
        self.load_timed()






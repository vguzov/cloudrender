import numpy as np
import glm
from OpenGL import GL as gl
from typing import List
from .camera import StandardProjectionCameraModel

class ShadowMap():
    def __init__(self, tracked_objects: List['Renderable'], camera: StandardProjectionCameraModel,
                 shadowmap_size: List[int]):
        self.tracked_objects = tracked_objects
        self.camera = camera
        self.shadowmap_size = shadowmap_size
        self._init()

    def _init(self):
        vport_params = np.zeros(4, dtype=np.int32)
        gl.glGetIntegerv(gl.GL_VIEWPORT, vport_params)
        self.original_viewport_size = vport_params[2:]
        gl.glViewport(0, 0, *self.shadowmap_size)
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, *self.shadowmap_size,
                     0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.texture, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    # def upload(self, tracked_object: Renderable):
    #     model_mtx = tracked_object.context.Model
    #     light_MVP = self.camera.context.Projection * self.camera.context.View * model_mtx
    #     shader_ids = tracked_object.shadowgen_context.shader_ids
    #     gl.glUniformMatrix4fv(shader_ids['light_MVP'], 1, gl.GL_FALSE,
    #                           glm.value_ptr(light_MVP))
    #     tracked_object._upload_uniforms(shader_ids)

    def update_shadowmap(self):
        gl.glViewport(0, 0, *self.shadowmap_size)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        for tracked_object in self.tracked_objects:
            tracked_object.draw_shadowmap(self.camera)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, *self.original_viewport_size)

    @property
    def lightVP(self):
        return self.camera.context.Projection * self.camera.context.View

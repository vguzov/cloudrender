import numpy as np
from OpenGL import GL as gl
from typing import List, Sequence
from ..camera.models import StandardProjectionCameraModel

class ShadowMap():
    def __init__(self, camera: StandardProjectionCameraModel,
                 shadowmap_size: Sequence[int]):
        self.camera = camera
        self.shadowmap_size = shadowmap_size
        self._init()

    def _remember_fbo(self):
        vport_params = np.zeros(4, dtype=np.int32)
        gl.glGetIntegerv(gl.GL_VIEWPORT, vport_params)
        self.original_viewport_size = vport_params[2:]
        fbo = np.zeros(1, dtype=np.int32)
        gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING, fbo)
        self.prev_draw_fbo = int(fbo)
        fbo = np.zeros(1, dtype=np.int32)
        gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING, fbo)
        self.prev_read_fbo = int(fbo)

    def _restore_fbo(self):
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.prev_draw_fbo)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.prev_read_fbo)
        gl.glViewport(0, 0, *self.original_viewport_size)

    def _init(self):
        self._remember_fbo()
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
        self._restore_fbo()

    # def upload(self, tracked_object: Renderable):
    #     model_mtx = tracked_object.context.Model
    #     light_MVP = self.camera.context.Projection * self.camera.context.View * model_mtx
    #     shader_ids = tracked_object.shadowgen_context.shader_ids
    #     gl.glUniformMatrix4fv(shader_ids['light_MVP'], 1, gl.GL_FALSE,
    #                           glm.value_ptr(light_MVP))
    #     tracked_object._upload_uniforms(shader_ids)

    def update_shadowmap(self, tracked_objects):
        self._remember_fbo()
        gl.glViewport(0, 0, *self.shadowmap_size)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        for tracked_object in tracked_objects:
            tracked_object.draw_shadowmap(self.camera)
        self._restore_fbo()

    @property
    def light_VP(self):
        return self.camera.context.Projection * self.camera.context.View

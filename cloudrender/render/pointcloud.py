import numpy as np
import os
import trimesh.points
from abc import ABC
from typing import List, Union
from dataclasses import dataclass

from OpenGL import GL as gl
from .renderable import Renderable
from .shaders.shader_loader import Shader
from ..camera.models import BaseCameraModel, StandardProjectionCameraModel
from .shadowmap import ShadowMap
from .lights import Light

class Pointcloud(Renderable, ABC):
    """
    Abstract class for all pointcloud objects
    """
    @dataclass
    class PointcloudContainer:
        vertices: np.ndarray
        colors: np.ndarray

    def __init__(self, camera: BaseCameraModel = None, draw_shadows: bool = True, generate_shadows: bool = True):
        super().__init__(camera, draw_shadows, generate_shadows)
        self.render_back = True

class SimplePointcloud(Pointcloud):
    """
    Pointcloud with simple rendering algorithm: one point - one color (only ambient lightning)
    """
    def __init__(self, *args, **kwargs):
        """
        Args:
            camera (BaseCameraModel): main camera
            shadowmaps (List[ShadowMap]): list of shadowmaps (no more than Renderable.SHADOWMAPS_MAX)
            additional_lights: list of lights
        """
        super().__init__(*args, **kwargs)

    def _init_shaders(self, camera_model, shader_mode):
        self.shader = shader = Shader()
        dirname = os.path.dirname(os.path.abspath(__file__))
        if self.draw_shadows:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/simple_pointcloud/shadowdraw/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, "shaders/simple_pointcloud/shadowdraw/fragment.glsl")],
                                      [os.path.join(dirname, "shaders/simple_pointcloud/shadowdraw/geometry.glsl")])
            self.context.shader_ids.update(self.locate_uniforms(self.shader, ['shadowmap_MVP', 'shadowmap_enabled',
                                                      'shadowmaps', 'shadow_color']))
        else:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/simple_pointcloud/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, "shaders/simple_pointcloud/fragment.glsl")],
                                      [os.path.join(dirname, "shaders/simple_pointcloud/geometry.glsl")])
        self.context.shader_ids.update(self.locate_uniforms(self.shader, ['splat_size']))

        if self.generate_shadows:
            shadowgen_shader = self.shadowgen_shader = Shader()
            shadowgen_shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/simple_pointcloud/shadowgen/vertex_{camera_model}.glsl")],
                                                [os.path.join(dirname,
                                                              "shaders/simple_pointcloud/shadowgen/fragment.glsl")],
                                                [os.path.join(dirname,
                                                              "shaders/simple_pointcloud/shadowgen/geometry.glsl")])
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader, ['splat_size']))

    def _finalize_init(self):
        self.set_splat_size(0.5)

    def _delete_buffers(self):
        gl.glDeleteBuffers(2, [self.context.vertexbuffer, self.context.colorbuffer])
        gl.glDeleteVertexArrays(1, [self.context.vao])

    def _set_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        assert len(glverts)==len(glcolors), "PC vertices and colors length should match"

        self.nglverts = len(glverts)

        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts,  gl.GL_STATIC_DRAW)

        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer( gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData( gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors,  gl.GL_STATIC_DRAW)

    def _update_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        gl.glBindVertexArray(self.context.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)

    def set_splat_size(self, splat_size):
        self.context.splat_size = splat_size

    def get_splat_size(self):
        return self.context.splat_size

    def _upload_uniforms(self, shader_ids, lights=(), shadowmaps=()):
        gl.glUniform1f(shader_ids['splat_size'], self.context.splat_size)
        shadowmaps_enabled = np.zeros(self.SHADOWMAPS_MAX, dtype=np.int32)
        shadowmaps_enabled[:len(shadowmaps)] = 1
        M = self.context.Model
        shadowmaps_lightMVP = [np.array(s.light_VP*M) for s in shadowmaps]
        shadowmaps_lightMVP = np.array(shadowmaps_lightMVP, dtype='f4')
        if self.draw_shadows:
            gl.glUniform1iv(self.context.shader_ids['shadowmap_enabled'], self.SHADOWMAPS_MAX, shadowmaps_enabled)
            gl.glUniformMatrix4fv(self.context.shader_ids['shadowmap_MVP'], len(shadowmaps), gl.GL_TRUE, shadowmaps_lightMVP)
            gl.glUniform4f(self.context.shader_ids['shadow_color'], *self.shadowcolor)
            for shadow_ind, shadowmap in enumerate(shadowmaps):
                gl.glActiveTexture(gl.GL_TEXTURE0+shadow_ind)
                gl.glBindTexture(gl.GL_TEXTURE_2D, shadowmap.texture)

    def _upload_shadowngen_uniforms(self, shader_ids):
        gl.glUniform1f(shader_ids['splat_size'], self.context.splat_size)

    def _draw(self, reset: bool, lights: List[Light], shadowmaps: List[ShadowMap]) -> bool:
        """
        Internal draw pass
        Args:
            reset (bool): Reset drawing progress (for progressive drawing)
            lights (List[Light]): All light objects that influence the current object
            shadowmaps (List[ShadowMap]): List of shadowmaps to draw shadows from
        Returns:
            bool: if drawing buffer was changed (if something was actually drawn)
        """
        if not reset:
            return False
        if not self.render_back:
            if np.array(self.context.MV).dot(np.array([0, 0, 1, 0]))[2] <= 0:
                return False
        self.shader.begin()
        self.upload_uniforms(self.context.shader_ids, lights, shadowmaps)

        gl.glBindVertexArray(self.context.vao)

        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        self.shader.end()
        return True

    def _draw_shadowmap(self, shadowmap_camera: StandardProjectionCameraModel) -> bool:
        """
        Shadow map draw pass - just to get depthmap values
        Args:
            shadowmap_camera (StandardProjectionCameraModel): perspective/ortho camera for shadow calculation
        Returns:
            bool: if drawing buffer was changed (if something was actually drawn)
        """
        self.shadowgen_shader.begin()
        self.upload_shadowgen_uniforms(shadowmap_camera, self.shadowgen_context.shader_ids)

        gl.glBindVertexArray(self.context.vao)

        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        self.shadowgen_shader.end()
        return True

class SimplePointcloudProgressive(SimplePointcloud):
    """
    SimplePointcloud with progressive drawing support
    """
    def __init__(self, *args, progressive_draw_size: int = None, progressive_draw_shuffle: bool = False, **kwargs):
        """
        Args:
            camera (BaseCameraModel): main camera
            shadowmaps (List[ShadowMap]): list of shadowmaps (no more than Renderable.SHADOWMAPS_MAX)
            additional_lights: list of lights
            progressive_draw_size (int): number of points draw in one pass, None for all
            progressive_draw_shuffle (bool): whether to shuffle drawing order during pointcloud load
        """
        super().__init__(*args, **kwargs)
        self.progressive_draw_size = progressive_draw_size
        self.progressive_draw_shuffle = progressive_draw_shuffle
        self.is_progressive = True
        self.current_offset = 0

    def _generate_indices(self, verts_count):
        inds = np.arange(verts_count, dtype=np.uint32)
        if self.progressive_draw_shuffle:
            np.random.shuffle(inds)
        return inds

    def _set_buffers(self, pointcloud: Pointcloud.PointcloudContainer):
        if self.progressive_draw_shuffle:
            inds = self._generate_indices(len(pointcloud.vertices))
            pointcloud = self.PointcloudContainer(pointcloud.vertices[inds], pointcloud.colors[inds])
        super()._set_buffers(pointcloud)
        self.current_offset = 0

    def _update_buffers(self, pointcloud: Pointcloud.PointcloudContainer):
        if self.progressive_draw_shuffle:
            inds = self._generate_indices(len(pointcloud.vertices))
            pointcloud = self.PointcloudContainer(pointcloud.vertices[inds], pointcloud.colors[inds])
        super()._update_buffers(pointcloud)
        self.current_offset = 0

    def _draw(self, reset: bool, lights: List[Light], shadowmaps: List[ShadowMap]) -> bool:
        if not self.render_back:
            if np.array(self.context.MV).dot(np.array([0, 0, 1, 0]))[2] <= 0:
                return False
        if reset:
            self.current_offset = 0
        if self.current_offset >= self.nglverts:
            return False
        self.shader.begin()
        self.upload_uniforms(self.context.shader_ids, lights, shadowmaps)

        gl.glBindVertexArray(self.context.vao)

        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        if self.progressive_draw_size is None:
            gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)
            self.current_offset += self.nglverts
        else:
            curr_len = min(self.progressive_draw_size, self.nglverts - self.current_offset)
            gl.glDrawArrays(gl.GL_POINTS, self.current_offset, curr_len)
            self.current_offset += self.progressive_draw_size

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        self.shader.end()
        return True

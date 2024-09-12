from pathlib import Path

import numpy as np
import os
import trimesh.points
from abc import ABC
from typing import List, Union, Optional
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
        normals: Optional[np.ndarray] = None

    def __init__(self, camera: BaseCameraModel = None, draw_shadows: bool = True, generate_shadows: bool = True):
        super().__init__(camera, draw_shadows, generate_shadows)
        self.render_back = True


class SimplePointcloud(Pointcloud):
    """
    Pointcloud with simple rendering algorithm: one point - one color (only ambient lightning)
    """

    def __init__(self, *args, indexing_offset: int = None, **kwargs):
        """
        Args:
            camera (BaseCameraModel): main camera
            shadowmaps (List[ShadowMap]): list of shadowmaps (no more than Renderable.SHADOWMAPS_MAX)
            additional_lights: list of lights
            indexing_offset: if not None, will render additional index map: each pixel has a projected vertex index.
                Index is computed as (vertex_id + indexing_offset)
        """
        super().__init__(*args, **kwargs)
        self.indexing_offset = indexing_offset
        self.shader_set_name = "simple_pointcloud"
        self.set_overlay_color()
        self.set_hsv_multiplier()

    def _init_shaders(self, camera_model, shader_mode):
        if self.indexing_offset is not None and (self.draw_shadows or self.generate_shadows):
            raise NotImplementedError("Rendering to index map is not yet supported with any shadowing")
        self.shader = shader = Shader()
        dirname = os.path.dirname(os.path.abspath(__file__))
        if self.draw_shadows:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/fragment.glsl")],
                                      [os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/geometry.glsl")])
            self.context.shader_ids.update(self.locate_uniforms(self.shader, ['shadowmap_MVP', 'shadowmap_enabled',
                                                                              'shadowmaps', 'shadow_color']))
        else:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/{self.shader_set_name}/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, f"shaders/{self.shader_set_name}/fragment.glsl")],
                                      [os.path.join(dirname, f"shaders/{self.shader_set_name}/geometry.glsl")])
        self.context.shader_ids.update(self.locate_uniforms(self.shader, ['splat_size', 'overlay_color', 'hsv_multiplier']))

        if self.generate_shadows:
            shadowgen_shader = self.shadowgen_shader = Shader()
            shadowgen_shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowgen/vertex_perspective.glsl")],
                                                [os.path.join(dirname,
                                                              f"shaders/{self.shader_set_name}/shadowgen/fragment.glsl")],
                                                [os.path.join(dirname,
                                                              f"shaders/{self.shader_set_name}/shadowgen/geometry.glsl")])
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader, ['splat_size']))

    def _finalize_init(self):
        self.set_splat_size(0.5)

    def _delete_buffers(self):
        gl.glDeleteBuffers(2, [self.context.vertexbuffer, self.context.colorbuffer])
        gl.glDeleteVertexArrays(1, [self.context.vao])

    def _set_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        assert len(glverts) == len(glcolors), "PC vertices and colors length should match"

        self.nglverts = len(glverts)

        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_STATIC_DRAW)

        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_STATIC_DRAW)

        if self.indexing_offset is not None:
            glids = np.arange(self.indexing_offset, self.indexing_offset + self.nglverts, dtype=np.int32)
            self.context.idbuffer = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, glids.nbytes, glids, gl.GL_STATIC_DRAW)

    def _update_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        self.nglverts = len(glverts)
        gl.glBindVertexArray(self.context.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)
        if self.indexing_offset is not None:
            glids = np.arange(self.indexing_offset, self.indexing_offset + self.nglverts, dtype=np.int32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, glids.nbytes, glids, gl.GL_DYNAMIC_DRAW)

    def set_splat_size(self, splat_size):
        self.context.splat_size = splat_size

    def set_overlay_color(self, color=(200, 200, 200, 0)):
        self.overlay_color = np.asarray(color, dtype=np.uint8)

    def set_hsv_multiplier(self, hsv_multiplier=(1, 1, 1)):
        self.hsv_multiplier = hsv_multiplier

    def get_splat_size(self):
        return self.context.splat_size

    def _upload_uniforms(self, shader_ids, lights=(), shadowmaps=()):
        gl.glUniform1f(shader_ids['splat_size'], self.context.splat_size)
        shadowmaps_enabled = np.zeros(self.SHADOWMAPS_MAX, dtype=np.int32)
        shadowmaps_enabled[:len(shadowmaps)] = 1
        M = self.context.Model
        shadowmaps_lightMVP = [np.array(s.light_VP * M) for s in shadowmaps]
        shadowmaps_lightMVP = np.array(shadowmaps_lightMVP, dtype='f4')
        gl.glUniform4f(self.context.shader_ids['overlay_color'], *(self.overlay_color.astype(np.float32) / 255.))
        gl.glUniform3f(self.context.shader_ids['hsv_multiplier'], *self.hsv_multiplier)
        if self.draw_shadows:
            # TODO: test if it should be shader_ids instead of self.context.shader_ids
            gl.glUniform1iv(self.context.shader_ids['shadowmap_enabled'], self.SHADOWMAPS_MAX, shadowmaps_enabled)
            gl.glUniformMatrix4fv(self.context.shader_ids['shadowmap_MVP'], len(shadowmaps), gl.GL_TRUE, shadowmaps_lightMVP)
            gl.glUniform4f(self.context.shader_ids['shadow_color'], *self.shadowcolor)
            for shadow_ind, shadowmap in enumerate(shadowmaps):
                gl.glActiveTexture(gl.GL_TEXTURE0 + shadow_ind)
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

        if self.indexing_offset is not None:
            gl.glEnableVertexAttribArray(2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glVertexAttribIPointer(2, 1, gl.GL_INT, 0, None)

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


class AvgcolorPointcloudWithNormals(Pointcloud):
    """
        Pointcloud with per-pixel color averaging from the nearest points (only ambient lightning)
    """

    def __init__(self, *args, resolution, **kwargs):
        """
        Args:
            camera (BaseCameraModel): main camera
            shadowmaps (List[ShadowMap]): list of shadowmaps (no more than Renderable.SHADOWMAPS_MAX)
            additional_lights: list of lights
        """
        super().__init__(*args, **kwargs)
        self.shader_set_name = "avgcolor_pointcloud_with_normals"
        self.resolution = resolution

    def _init_shaders(self, camera_model, shader_mode):
        self.agg_shader = Shader()
        self.norm_shader = Shader()
        self.dmap_shader = Shader()

        dirname = Path(os.path.dirname(os.path.abspath(__file__)))
        if self.draw_shadows:
            # shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/vertex_{camera_model}.glsl")],
            #                           [os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/fragment.glsl")],
            #                           [os.path.join(dirname, f"shaders/{self.shader_set_name}/shadowdraw/geometry.glsl")])
            # self.context.shader_ids.update(self.locate_uniforms(self.shader, ['shadowmap_MVP', 'shadowmap_enabled',
            #                                                                   'shadowmaps', 'shadow_color']))

            # TODO: shadows support
            self.agg_shader.initShaderFromGLSL([dirname / f"shaders/{self.shader_set_name}/vertex_{camera_model}.glsl"],
                                               [dirname / f"shaders/{self.shader_set_name}/fragment.glsl"],
                                               [dirname / f"shaders/{self.shader_set_name}/geometry.glsl"])
        else:
            self.agg_shader.initShaderFromGLSL([dirname / f"shaders/{self.shader_set_name}/vertex_{camera_model}.glsl"],
                                               [dirname / f"shaders/{self.shader_set_name}/fragment.glsl"],
                                               [dirname / f"shaders/{self.shader_set_name}/geometry.glsl"])
        self.norm_shader.initShaderFromGLSL([dirname / f"shaders/{self.shader_set_name}/normalization/vertex.glsl"],
                                            [dirname / f"shaders/{self.shader_set_name}/normalization/fragment.glsl"])
        self.dmap_shader.initShaderFromGLSL([dirname / f"shaders/{self.shader_set_name}/depthmap/vertex_{camera_model}.glsl"],
                                            [dirname / f"shaders/{self.shader_set_name}/depthmap/fragment.glsl"],
                                            [dirname / f"shaders/{self.shader_set_name}/depthmap/geometry.glsl"])
        self.context.agg_shader_ids = {}
        self.context.dmap_shader_ids = {}
        self.context.norm_shader_ids = {}
        self.context.agg_shader_ids.update(self.locate_uniforms(self.agg_shader, ['splat_size']))
        self.context.dmap_shader_ids.update(self.locate_uniforms(self.dmap_shader, ['splat_size', 'depth_offset']))
        self.context.norm_shader_ids.update(self.locate_uniforms(self.norm_shader, ['pixColors', 'pixWeights', 'resolution']))

        if self.generate_shadows:
            shadowgen_shader = self.shadowgen_shader = Shader()
            shadowgen_shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/{self.shader_set_name}/depthmap/vertex_perspective.glsl")],
                                                [os.path.join(dirname,
                                                              f"shaders/{self.shader_set_name}/depthmap/fragment.glsl")],
                                                [os.path.join(dirname,
                                                              f"shaders/{self.shader_set_name}/depthmap/geometry.glsl")])
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader, ['splat_size']))

    def _reload_shaders(self, shader_mode: str = None):
        self._init_shaders(self.camera.model, shader_mode if shader_mode is not None else self.shader_mode)
        self.context.agg_shader_ids.update(self.camera.locate_uniforms(self.agg_shader))
        self.context.agg_shader_ids.update(self.locate_uniforms(self.agg_shader, ["M"]))
        self.context.dmap_shader_ids.update(self.camera.locate_uniforms(self.dmap_shader))
        self.context.dmap_shader_ids.update(self.locate_uniforms(self.dmap_shader, ["M"]))
        if self.generate_shadows:
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader,
                                                                          StandardProjectionCameraModel.uniforms_names))
            self.shadowgen_context.shader_ids.update(self.locate_uniforms(self.shadowgen_shader, ["M"]))

    def _finalize_init(self):
        self.set_splat_size(0.5)
        self.set_depth_acc_offset(1e-3)

    def _create_aggregation_fbo(self):
        width, height = self.resolution
        binded_fb = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)

        fbo = self.context.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo)

        coltex = self.context.agg_color_texture = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, coltex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture(gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, coltex, 0)

        self.context.depthbuf = gl.glGenRenderbuffers(1)
        # gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._seqdraw_cb)
        # gl.glRenderbufferStorage(
        #     gl.GL_RENDERBUFFER, gl.GL_RGBA,
        #     width, height
        # )
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.context.depthbuf)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT32,
            width, height
        )
        # gl.glFramebufferRenderbuffer(
        #     gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
        #     gl.GL_RENDERBUFFER, self._seqdraw_cb
        # )
        gl.glFramebufferRenderbuffer(
            gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER, self.context.depthbuf
        )
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, binded_fb)

    def _activate_aggregation_fbo(self):
        self.binded_fb = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.context.fbo)

    def _deactivate_aggregation_fbo(self):
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.binded_fb)

    def _destroy_aggregation_fbo(self):
        self._activate_aggregation_fbo()
        gl.glDeleteRenderbuffers(1, [self.context.depthbuf])
        gl.glDeleteTextures(1, [self.context.agg_color_texture])
        self._deactivate_aggregation_fbo()
        gl.glDeleteFramebuffers(1, [self.context.fbo])
        self.context.fbo = None
        self.context.depthbuf = None
        self.context.agg_color_texture = None

    def _get_glnorms(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        if isinstance(pointcloud, trimesh.points.PointCloud):
            assert "ply_raw" in pointcloud.metadata and "nx" in pointcloud.metadata['ply_raw']['vertex'][
                'properties'], "PC normals are required for this type of cloud"
            glnorms = np.copy(np.stack([pointcloud.metadata['ply_raw']['vertex']['data'][x] for x in ["nx", "ny", "nz"]], axis=1).astype(np.float32),
                              order='C')
        else:
            assert pointcloud.normals is not None, "PC normals are required for this type of cloud"
            glnorms = np.copy(pointcloud.normals.astype(np.float32), order='C')
        return glnorms

    def _set_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        self._create_aggregation_fbo()
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        assert len(glverts) == len(glcolors), "PC vertices and colors length should match"
        glnorms = self._get_glnorms(pointcloud)

        self.nglverts = len(glverts)

        self._activate_aggregation_fbo()

        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_STATIC_DRAW)

        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_STATIC_DRAW)

        self.context.normsbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_STATIC_DRAW)

        self._deactivate_aggregation_fbo()

        self.context.coords_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.coords_vao)

        glcoords = np.copy(np.dstack(np.meshgrid(*[np.arange(x) for x in self.resolution])).reshape(-1, 2).astype(np.float32), order='C')
        self.context.coordsbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.coordsbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcoords.nbytes, glcoords, gl.GL_STATIC_DRAW)

    def _update_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        glnorms = self._get_glnorms(pointcloud)
        self.nglverts = len(glverts)
        self._activate_aggregation_fbo()
        gl.glBindVertexArray(self.context.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_DYNAMIC_DRAW)
        self._deactivate_aggregation_fbo()

    def _delete_buffers(self):
        gl.glDeleteBuffers(4, [self.context.vertexbuffer, self.context.colorbuffer, self.context.normsbuffer, self.context.coordsbuffer])
        gl.glDeleteVertexArrays(2, [self.context.vao, self.context.coords_vao])
        self._destroy_aggregation_fbo()

    def set_splat_size(self, splat_size):
        self.context.splat_size = splat_size

    def get_splat_size(self):
        return self.context.splat_size

    def set_depth_acc_offset(self, depth_offset):
        self.context.depth_offset = depth_offset

    def get_depth_acc_offset(self):
        return self.context.depth_offset

    def _upload_uniforms(self, shader_ids, lights=(), shadowmaps=()):
        gl.glUniform1f(shader_ids['splat_size'], self.context.splat_size)
        if "depth_offset" in shader_ids:
            gl.glUniform1f(shader_ids['depth_offset'], self.context.depth_offset)
        if self.draw_shadows:
            return
            # TODO: make shadows support
            shadowmaps_enabled = np.zeros(self.SHADOWMAPS_MAX, dtype=np.int32)
            shadowmaps_enabled[:len(shadowmaps)] = 1
            M = self.context.Model
            shadowmaps_lightMVP = [np.array(s.light_VP * M) for s in shadowmaps]
            shadowmaps_lightMVP = np.array(shadowmaps_lightMVP, dtype='f4')
            gl.glUniform1iv(shader_ids['shadowmap_enabled'], self.SHADOWMAPS_MAX, shadowmaps_enabled)
            gl.glUniformMatrix4fv(shader_ids['shadowmap_MVP'], len(shadowmaps), gl.GL_TRUE, shadowmaps_lightMVP)
            gl.glUniform4f(shader_ids['shadow_color'], *self.shadowcolor)
            for shadow_ind, shadowmap in enumerate(shadowmaps):
                gl.glActiveTexture(gl.GL_TEXTURE0 + shadow_ind)
                gl.glBindTexture(gl.GL_TEXTURE_2D, shadowmap.texture)

    def upload_normshader_uniforms(self, shader_ids):
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.context.agg_color_texture)
        gl.glUniform2f(shader_ids["resolution"], *self.resolution)

    def _upload_shadowngen_uniforms(self, shader_ids):
        gl.glUniform1f(shader_ids['splat_size'], self.context.splat_size)
        gl.glUniform1f(shader_ids['depth_offset'], 0.)

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

        self._activate_aggregation_fbo()
        gl.glViewport(0, 0, *self.resolution)
        prev_clear_color = gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE)
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LESS)

        # Pass 1: depthmap

        self.dmap_shader.begin()
        self.upload_uniforms(self.context.dmap_shader_ids, lights, shadowmaps)

        gl.glBindVertexArray(self.context.vao)

        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
        self.dmap_shader.end()

        # Pass 2: color aggregation

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)

        self.agg_shader.begin()
        self.upload_uniforms(self.context.agg_shader_ids, lights, shadowmaps)

        gl.glBindVertexArray(self.context.vao)

        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glEnableVertexAttribArray(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
        self.agg_shader.end()

        self._deactivate_aggregation_fbo()

        gl.glDepthMask(gl.GL_TRUE)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(*prev_clear_color)

        # Pass 3: picture normalization
        # TODO: proper depth handling
        self.norm_shader.begin()
        self.upload_normshader_uniforms(self.context.norm_shader_ids)
        gl.glBindVertexArray(self.context.coords_vao)
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.coordsbuffer)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.resolution[0] * self.resolution[1])

        gl.glDisableVertexAttribArray(0)
        self.norm_shader.end()

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

        gl.glEnableVertexAttribArray(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
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

        if self.indexing_offset is not None:
            gl.glEnableVertexAttribArray(2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glVertexAttribIPointer(2, 1, gl.GL_INT, 0, None)

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


class SimplePointcloudWithNormals(SimplePointcloud):
    """
        SimplePointcloud with splats oriented according to per-point normals
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shader_set_name = "simple_pointcloud_with_normals"

    def _get_glnorms(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        if isinstance(pointcloud, trimesh.points.PointCloud):
            assert "ply_raw" in pointcloud.metadata and "nx" in pointcloud.metadata['ply_raw']['vertex'][
                'properties'], "PC normals are required for this type of cloud"
            glnorms = np.copy(np.stack([pointcloud.metadata['ply_raw']['vertex']['data'][x] for x in ["nx", "ny", "nz"]], axis=1).astype(np.float32),
                              order='C')
        else:
            assert pointcloud.normals is not None, "PC normals are required for this type of cloud"
            glnorms = np.copy(pointcloud.normals.astype(np.float32), order='C')
        return glnorms

    def _set_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        assert len(glverts) == len(glcolors), "PC vertices and colors length should match"
        glnorms = self._get_glnorms(pointcloud)

        self.nglverts = len(glverts)

        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_STATIC_DRAW)

        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_STATIC_DRAW)

        self.context.normsbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_STATIC_DRAW)

        if self.indexing_offset is not None:
            glids = np.arange(self.indexing_offset, self.indexing_offset + self.nglverts, dtype=np.int32)
            self.context.idbuffer = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, glids.nbytes, glids, gl.GL_STATIC_DRAW)

    def _update_buffers(self, pointcloud: Union[Pointcloud.PointcloudContainer, trimesh.points.PointCloud]):
        glverts = np.copy(pointcloud.vertices.astype(np.float32), order='C')
        glcolors = np.copy(pointcloud.colors.astype(np.float32) / 255., order='C')
        glnorms = self._get_glnorms(pointcloud)
        self.nglverts = len(glverts)
        gl.glBindVertexArray(self.context.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_DYNAMIC_DRAW)
        if self.indexing_offset is not None:
            glids = np.arange(self.indexing_offset, self.indexing_offset + self.nglverts, dtype=np.int32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, glids.nbytes, glids, gl.GL_DYNAMIC_DRAW)

    def _delete_buffers(self):
        gl.glDeleteBuffers(3, [self.context.vertexbuffer, self.context.colorbuffer, self.context.normsbuffer])
        gl.glDeleteVertexArrays(1, [self.context.vao])

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

        gl.glEnableVertexAttribArray(3)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        if self.indexing_offset is not None:
            gl.glEnableVertexAttribArray(2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.idbuffer)
            gl.glVertexAttribIPointer(2, 1, gl.GL_INT, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(3)
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

        gl.glEnableVertexAttribArray(3)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normsbuffer)
        gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(3)
        self.shadowgen_shader.end()
        return True

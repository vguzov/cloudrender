import os

import logging
import numpy as np
from abc import ABC
from typing import List, Union
from dataclasses import dataclass

import trimesh
from OpenGL import GL as gl
from .shaders.shader_loader import Shader
from .renderable import Renderable
from .lights import Light, DirectionalLight
from .shadowmap import ShadowMap
from ..camera.models import StandardProjectionCameraModel


class Mesh(Renderable, ABC):
    @dataclass
    class MeshContainer:
        vertices: np.ndarray
        faces: np.ndarray
        colors: np.ndarray
        vertex_normals: np.ndarray

    def __init__(self, *args, draw_shadows: bool = True, generate_shadows: bool = True, **kwargs):
        super().__init__(*args, draw_shadows=draw_shadows, generate_shadows=generate_shadows, **kwargs)

class SimpleMesh(Mesh):
    @dataclass
    class MaterialProps:
        ambient: float = 1.
        diffuse: float = 0.
        specular: float = 0.
        shininess: float = 0.
    """
    Vertex-colored mesh with directional lighting support
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.material = SimpleMesh.MaterialProps()

    def _init_shaders(self, camera_model, shader_mode):
        self.shader = shader = Shader()
        dirname = os.path.dirname(os.path.abspath(__file__))

        if self.draw_shadows:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/simple_mesh/shadowdraw/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, "shaders/simple_mesh/shadowdraw/fragment.glsl")])
            self.context.shader_ids.update(self.locate_uniforms(self.shader, ['shadowmap_MVP', 'shadowmap_enabled',
                                                      'shadowmaps', 'shadow_color']))
        else:
            shader.initShaderFromGLSL([os.path.join(dirname, f"shaders/simple_mesh/vertex_{camera_model}.glsl")],
                                      [os.path.join(dirname, "shaders/simple_mesh/fragment.glsl")])
        self.context.shader_ids.update(self.locate_uniforms(self.shader, ['dirlight.direction', 'dirlight.intensity',
                                                                          'specular', 'shininess',
                                                                          'ambient', 'diffuse']))

        if self.generate_shadows:
            self.shadowgen_shader = Shader()
            self.shadowgen_shader.initShaderFromGLSL([
                os.path.join(dirname, f"shaders/simple_mesh/shadowgen/vertex_{camera_model}.glsl")],
                [os.path.join(dirname, "shaders/simple_mesh/shadowgen/fragment.glsl")])

    def _delete_buffers(self):
        gl.glDeleteBuffers(3, [self.context.vertexbuffer, self.context.colorbuffer, self.context.normalbuffer])
        gl.glDeleteVertexArrays(1, [self.context.vao])

    def set_material(self, ambient=1., diffuse=0., specular=0., shininess=0.):
        self.material = self.MaterialProps(ambient, diffuse, specular, shininess)

    def _set_buffers(self, mesh: Union[Mesh.MeshContainer, trimesh.Trimesh]):
        faces = mesh.faces
        glverts = np.copy(mesh.vertices.astype(np.float32)[faces.reshape(-1), :], order='C')
        if hasattr(mesh, "colors"):
            glcolors = np.copy(mesh.colors.astype(np.float32)[faces.reshape(-1), :] / 255., order='C')
        else:
            glcolors = np.copy(mesh.visual.vertex_colors.astype(np.float32)[faces.reshape(-1), :] / 255., order='C')
        assert glcolors.shape[1] == 4
        glnorms = np.copy(mesh.vertex_normals.astype(np.float32)[faces.reshape(-1), :], order='C')

        self.nglverts = len(glverts)

        self.context.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.context.vao)

        self.context.vertexbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)

        self.context.colorbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)

        self.context.normalbuffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normalbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_DYNAMIC_DRAW)

    def _update_buffers(self, mesh: Union[Mesh.MeshContainer, trimesh.Trimesh]):
        faces = mesh.faces
        glverts = np.copy(mesh.vertices.astype(np.float32)[faces.reshape(-1), :], order='C')
        if hasattr(mesh, "colors"):
            glcolors = np.copy(mesh.colors.astype(np.float32)[faces.reshape(-1), :] / 255., order='C')
        else:
            glcolors = np.copy(mesh.visual.vertex_colors.astype(np.float32)[faces.reshape(-1), :] / 255., order='C')
        assert glcolors.shape[1] == 4
        glnorms = np.copy(mesh.vertex_normals.astype(np.float32)[faces.reshape(-1), :], order='C')
        gl.glBindVertexArray(self.context.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.vertexbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glverts.nbytes, glverts, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.colorbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glcolors.nbytes, glcolors, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normalbuffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, glnorms.nbytes, glnorms, gl.GL_DYNAMIC_DRAW)

    def _upload_uniforms(self, shader_ids, lights=(), shadowmaps=()):
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
        if len(lights)>0:
            # currently only 1 directional light is supported
            light = lights[0]
            material = self.material
        else:
            # if no light is supplied, make the object fully ambient
            light = DirectionalLight(np.ones(3), np.ones(3))
            material = self.MaterialProps()
        gl.glUniform3f(self.context.shader_ids['dirlight.direction'], *light.direction)
        gl.glUniform3f(self.context.shader_ids['dirlight.intensity'], *light.intensity)
        gl.glUniform1f(self.context.shader_ids['ambient'], material.ambient)
        gl.glUniform1f(self.context.shader_ids['diffuse'], material.diffuse)
        gl.glUniform1f(self.context.shader_ids['specular'], material.specular)
        gl.glUniform1f(self.context.shader_ids['shininess'], material.shininess)

    def _draw(self, reset: bool, lights: List[Light], shadowmaps: List[ShadowMap]) -> bool:
        """
        Internal draw pass
        Args:
            reset (bool): Reset drawing progress (for progressive drawing)
            lights (List[Light]): All light objects that influence the current object
        Returns:
            bool: if drawing buffer was changed (if something was actually drawn)
        """
        if not reset:
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

        gl.glEnableVertexAttribArray(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.context.normalbuffer)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
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

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.nglverts)

        gl.glDisableVertexAttribArray(0)
        self.shadowgen_shader.end()
        return True





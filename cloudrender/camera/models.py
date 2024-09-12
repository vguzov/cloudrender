from typing import Sequence, Union, Optional

import numpy as np
from abc import ABC, abstractmethod
from OpenGL import GL as gl
import glm
import os
from scipy.spatial.transform import Rotation


class BaseCameraModel(ABC):
    uniforms_names = ["M", "V"]

    class CameraContext(object):
        pass

    def __init__(self, camera_model):
        self.context = self.CameraContext()
        self.model = camera_model
        self.context.View = glm.mat4(1.0)
        self.quat = np.array([1., 0, 0, 0])
        self.pose = np.zeros(3)
        self.world2cam = False

    def init_extrinsics(self, quat=None, pose=None, world2cam=False):
        quat = self.quat if quat is None else quat
        pose = self.pose if pose is None else pose
        self.quat = quat
        self.pose = pose
        self.world2cam = world2cam
        R = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
        t = np.array([pose]).T
        if world2cam:
            RT = np.vstack([np.hstack([R, t]), [[0, 0, 0, 1]]])
        else:
            # otherwise invert cam2world to world2cam
            RT = np.vstack([np.hstack([R.T, -np.matmul(R.T, t)]), [[0, 0, 0, 1]]])
        self.context.View = glm.mat4(*(RT.T.astype(np.float32).copy().flatten()))

    @abstractmethod
    def init_intrinsics(self, **kwargs):
        pass

    def upload_extrinsics(self, shader_ids):
        gl.glUniformMatrix4fv(shader_ids['V'], 1, gl.GL_FALSE, glm.value_ptr(self.context.View))

    @abstractmethod
    def upload_intrinsics(self, shader_ids):
        pass

    def upload(self, shader_ids):
        self.upload_extrinsics(shader_ids)
        self.upload_intrinsics(shader_ids)

    def locate_uniforms(self, shader, keys=None):
        keys = self.uniforms_names if keys is None else keys
        return {k: gl.glGetUniformLocation(shader.program, k) for k in keys}

    def project(self, points: np.ndarray, model_mtx: Optional[Union[glm.mat4x4, np.ndarray]] = None) -> np.ndarray:
        """
        Projects the points according to the camera model and returns the projected points in NDC (normalized device coordinates)
        Args:
            points (np.ndarray): points in the world coordinates

        Returns:
            np.ndarray: projected points in NDC, shape (-1,3)
        """
        raise NotImplementedError("Projection is not implemented for this camera type")


class OcamCameraModel(BaseCameraModel):
    uniforms_names = BaseCameraModel.uniforms_names + ['ocam_invpol', 'ocam_affine', 'ocam_center_off',
                                                       'ocam_theta_thresh', 'far', 'width_mul']

    def __init__(self):
        super().__init__("ocam")

    def init_intrinsics(self, cameramodel_dict, fov=360, far=20.):
        ocammodel_dict = cameramodel_dict['OCamModel']
        # polynomial coefficients for the DIRECT mapping function
        ocam_pol = [float(x) for x in ocammodel_dict['cam2world']['coeff']]
        # polynomial coefficients for the inverse mapping function
        ocam_invpol = np.array([float(x) for x in ocammodel_dict['world2cam']['coeff']])
        # center: "row" and "column", starting from 0 (C convention)
        ocam_xy_center = np.array((float(ocammodel_dict['cx']), float(ocammodel_dict['cy'])))
        # _affine parameters "c", "d", "e"
        ocam_affine = np.array([float(ocammodel_dict[x]) for x in ['c', 'd', 'e']])
        # image size: "height" and "width"
        ocam_imsize = cameramodel_dict['ImageSize']
        ocam_img_size = np.array((int(ocam_imsize['Width']), int(ocam_imsize['Height'])))

        self.context.ocam_invpol = ocam_invpol / ocam_img_size[0] * 2

        self.context.ocam_center_off = ocam_xy_center / ocam_img_size[::-1] * 2 - 1
        # self.context.ocam_center_off = (ocam_xy_center - ocam_img_size[::-1] / 2) / ocam_img_size * 2
        self.context.ocam_theta_thresh = np.deg2rad(fov / 2) - np.pi / 2
        self.context.ocam_affine = ocam_affine.copy()
        self.context.ocam_affine[:2] *= ocam_img_size[0] / ocam_img_size[1]
        self.context.far = far
        self.context.width_mul = ocam_img_size[1] / ocam_img_size[0]

    def upload_intrinsics(self, shader_ids):
        gl.glUniform1dv(shader_ids['ocam_invpol'], 18, self.context.ocam_invpol.astype(np.float64).copy())
        gl.glUniform3dv(shader_ids['ocam_affine'], 1, self.context.ocam_affine.astype(np.float64).copy())
        gl.glUniform2dv(shader_ids['ocam_center_off'], 1,
                        self.context.ocam_center_off.astype(np.float64).copy())
        gl.glUniform1f(shader_ids['ocam_theta_thresh'], float(self.context.ocam_theta_thresh))
        gl.glUniform1f(shader_ids['far'], float(self.context.far))
        gl.glUniform1f(shader_ids['width_mul'], self.context.width_mul)


class OpenCVCameraModel(BaseCameraModel):
    uniforms_names = BaseCameraModel.uniforms_names + ['distortion_coeff', 'center_off',
                                                       'focal_dist', 'far', 'width_mul']

    def __init__(self):
        super().__init__("opencv")

    def init_intrinsics(self, image_size, focal_dist, center, distortion_coeffs, far=20.):
        assert len(distortion_coeffs) == 5
        image_size = np.array(image_size)
        focal_dist = np.array(focal_dist)
        center = np.array(center)
        distortion_coeffs = np.array(distortion_coeffs)
        self.context.focal_dist = (focal_dist / image_size * 2).astype(np.float32).copy()
        self.context.center_off = (center / image_size * 2 - 1).astype(np.float32).copy()
        self.context.distortion_coeffs = distortion_coeffs.astype(np.float32).copy()
        self.context.far = np.array(far).astype(np.float32).copy()
        self.context.width_mul = image_size[1] / image_size[0]

    def upload_intrinsics(self, shader_ids):
        gl.glUniform1fv(shader_ids['distortion_coeff'], 5, self.context.distortion_coeffs)
        gl.glUniform2fv(shader_ids['center_off'], 1, self.context.center_off)
        gl.glUniform2fv(shader_ids['focal_dist'], 1, self.context.focal_dist)
        gl.glUniform1f(shader_ids['far'], self.context.far)
        gl.glUniform1f(shader_ids['width_mul'], self.context.width_mul)


class StandardProjectionCameraModel(BaseCameraModel, ABC):
    uniforms_names = BaseCameraModel.uniforms_names + ['P', 'width_mul']

    def __init__(self, name):
        super().__init__(name)

    def upload_intrinsics(self, shader_ids):
        gl.glUniformMatrix4fv(shader_ids['P'], 1, gl.GL_FALSE, glm.value_ptr(self.context.Projection))
        gl.glUniform1f(shader_ids['width_mul'], self.context.width_mul)

    def project(self, points: np.ndarray, model_mtx: Optional[Union[glm.mat4x4, np.ndarray]] = None):
        VP = np.asarray(self.context.Projection * self.context.View)
        if model_mtx is not None:
            VP = np.matmul(VP, np.asarray(model_mtx))
        points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        points_gl_projected = np.matmul(points, VP.T)
        points_gl_projected = points_gl_projected[:, :3] / points_gl_projected[:, 3]  # NDC
        return points_gl_projected


class PerspectiveCameraModel(StandardProjectionCameraModel):
    def __init__(self):
        super().__init__("perspective")

    def init_intrinsics(self, image_size, fov=45., far=20., near=0.05):
        width, height = image_size
        self.context.Projection = glm.perspective(glm.radians(fov), float(width) / float(height), near, far)
        self.context.width_mul = image_size[1] / image_size[0]


class OrthogonalCameraModel(StandardProjectionCameraModel):
    def __init__(self):
        super().__init__("orthogonal")

    def init_intrinsics(self, image_size, left, right, bottom, top, far=20., near=0.05):
        width, height = image_size
        self.context.Projection = glm.orthoLH(left, right, bottom, top, near, far)
        self.context.width_mul = image_size[1] / image_size[0]


camera_models = {'ocam': OcamCameraModel, 'opencv': OpenCVCameraModel, 'perspective': PerspectiveCameraModel}

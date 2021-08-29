#!/usr/bin/env python

# BSD 3-Clause License
#
# Copyright (c) 2018, Centre National de la Recherche Scientifique
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified by Vladimir Guzov, 2021

import OpenGL.EGL as egl
from .. import EGL_PLATFORM_DEVICE_EXT, EGL_DRM_DEVICE_FILE_EXT, egl_convert_to_int_array
from ctypes import pointer

class GenericEGLSurface:
    def __init__(self, egl_dpy, egl_config):
        self.egl_dpy, self.egl_config = egl_dpy, egl_config
    def initialize(self, width, height):
        pb_surf_attribs = egl_convert_to_int_array({
                egl.EGL_WIDTH: width,
                egl.EGL_HEIGHT: height,
        })
        self.egl_surface = egl.eglCreatePbufferSurface(
                self.egl_dpy, self.egl_config, pb_surf_attribs)
        if self.egl_surface == egl.EGL_NO_SURFACE:
            return False
        return True
    def release(self):
        egl.eglDestroySurface(self.egl_dpy, self.egl_surface)
    def make_current(self, egl_context):
        return egl.eglMakeCurrent(self.egl_dpy, self.egl_surface, self.egl_surface, egl_context)

class GenericEGLDevice:
    @staticmethod
    def probe():
        if not hasattr(egl, 'eglQueryDevicesEXT'):
            # if no enumeration support in EGL, return empty list
            return []
        num_devices = egl.EGLint()
        if not egl.eglQueryDevicesEXT(0, None, pointer(num_devices)) or num_devices.value < 1:
            return []
        devices = (egl.EGLDeviceEXT * num_devices.value)() # array of size num_devices
        if not egl.eglQueryDevicesEXT(num_devices.value, devices, pointer(num_devices)) or num_devices.value < 1:
            return []
        return [ GenericEGLDevice(devices[i]) for i in range(num_devices.value) ]
    def __init__(self, egl_dev):
        self.egl_dev = egl_dev
    def get_egl_display(self):
        return egl.eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, self.egl_dev, None)
    def initialize(self):
        return True
    def release(self):
        pass
    def compatible_surface_type(self):
        return egl.EGL_PBUFFER_BIT
    @property
    def name(self):
        if not hasattr(egl, 'eglQueryDeviceStringEXT'):
            return "<unknown EGL device>"
        devstr = egl.eglQueryDeviceStringEXT(self.egl_dev, EGL_DRM_DEVICE_FILE_EXT)
        if devstr is None:
            return "<unknown EGL device>"
        return "EGL device " + devstr.decode('ASCII')
    def create_surface(self, egl_dpy, egl_config):
        return GenericEGLSurface(egl_dpy, egl_config)

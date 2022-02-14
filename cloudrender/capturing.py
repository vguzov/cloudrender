import ctypes
import numpy as np
from OpenGL import GL as gl

class DirectCapture:
    """A helper capturing class. Gets the color or depth data from the current FBO"""
    def __init__(self, resolution):
        self.resolution = resolution

    def request_color(self):
        width, height = self.resolution
        color_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        color = np.frombuffer(color_buf, np.uint8).reshape(height, width, 3)
        color = color[::-1].copy()
        return color

    def request_depth(self):
        width, height = self.resolution
        depth_buf = gl.glReadPixels(0, 0, width, height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        depth = np.frombuffer(depth_buf, np.float32).reshape(height, width)
        depth = depth[::-1].copy()
        return depth

    def request_inds(self):
        width, height = self.resolution
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
        ind_buf = gl.glReadPixels(0, 0, width, height, gl.GL_RED_INTEGER, gl.GL_INT)
        indices = np.frombuffer(ind_buf, np.int32).reshape(height, width)[::-1]
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        return indices

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncPBOCapture:
    """A helper capturing class. Gets the color from the current FBO.
    Submits the task to the OpenGL driver to be executed asynchronously.
    Doesn't wait on CPU<->GPU exchange which improves speed in most cases"""
    def __init__(self, resolution, queue_size):
        self.queue_size = queue_size
        self.resolution = resolution
        self.qstart = 0
        self.qlen = 0
        self.pbos = None

    def create_pbos(self):
        pbos = gl.glGenBuffers(self.queue_size)
        for pbo in pbos:
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, (3 * self.resolution[0] * self.resolution[1]), None,
                            gl.GL_STREAM_READ)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        self.pbos = pbos

    def delete_pbos(self):
        if self.pbos is not None:
            gl.glDeleteBuffers(self.queue_size, self.pbos)
            self.pbos = None

    def __len__(self):
        return self.qlen

    def __enter__(self):
        self.create_pbos()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_pbos()

    def get_first_requested_color(self):
        if self.qlen == 0:
            return None
        width, height = self.resolution
        pbo = self.pbos[self.qstart]
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
        bufferdata = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
        data = np.frombuffer(ctypes.string_at(bufferdata, (3 * width * height)), np.uint8).reshape(height, width, 3)
        data = data[::-1].copy()
        gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        self.qlen -= 1
        self.qstart += 1
        if self.qstart >= self.queue_size:
            self.qstart = 0
        return data

    @property
    def qend(self):
        return (self.qstart+self.qlen) % self.queue_size

    def request_color_async(self):
        if self.qlen >= self.queue_size:
            res = self.get_first_requested_color()
        else:
            res = None
        pbo = self.pbos[self.qend]
        self.qlen += 1
        width, height = self.resolution
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, pbo)
        gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, 0)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        return res



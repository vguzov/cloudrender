import numpy as np
from videoio import Uint16Reader, VideoReader

from .pointcloud import SimplePointcloud
from .renderable import DynamicTimedRenderable


class DepthVideo(SimplePointcloud, DynamicTimedRenderable):
    VIDEO_RELOAD_THRESH = 100
    DEFAULT_COLOR = (255, 255, 0, 255)

    def __init__(self, pc_table, color = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_table_ext = np.dstack([pc_table, np.ones(pc_table.shape[:2] + (1,), dtype=pc_table.dtype)])
        self.color_override = False
        self.start_ind = 0
        self.current_sequence_frame_ind = -1
        self.seqlen = 0
        self.current_frame_cloud = None
        if color is None:
            self.color = self.DEFAULT_COLOR
        else:
            self.color = color

    def _set_sequence(self, depthvideo_path, colorvideo_path = None):
        self.depthvideo_path = depthvideo_path
        self.colorvideo_path = colorvideo_path
        self.dmaps_reader = Uint16Reader(depthvideo_path)
        self.dmaps_iter = iter(self.dmaps_reader)
        self.sequence_len = len(self.dmaps_reader)
        if colorvideo_path is not None:
            self.colors_reader = VideoReader(colorvideo_path)
            self.colors_iter = iter(self.colors_reader)
            self.sequence_len = min(self.sequence_len, len(self.colors_reader))
        else:
            self.colors_reader = None
            self.colors_iter = None
        self.next_frame_ind = 0
        self.reset_current_frame()

    def switch_color_override(self):
        self.color_override = not self.color_override
        if self.sequence_len != 0:
            self.load_current_frame()

    def unset_video(self):
        self.depthvideo_path = None
        self.dmaps_reader = None
        self.dmaps_iter = None
        self.colorvideo_path = None
        self.colors_reader = None
        self.colors_iter = None
        self.sequence_len = 0
        self.times = None
        self.current_frame_cloud = None
        self.current_sequence_frame_ind = self.start_ind
        self.delete_buffers()

    def _load_current_frame(self):
        pc = self.get_curr_pointcloud()
        self.update_buffers(pc)

    def load_depth_color(self, frame_ind):
        diff = frame_ind - self.next_frame_ind
        if diff < self.VIDEO_RELOAD_THRESH and diff>=0:
            for _ in range(diff):
                next(self.dmaps_iter)
                if self.colorvideo_path is not None:
                    next(self.colors_iter)
        else:
            self.dmaps_reader = Uint16Reader(self.depthvideo_path, start_frame=frame_ind)
            self.dmaps_iter = iter(self.dmaps_reader)
            if self.colorvideo_path is not None:
                self.colors_reader = VideoReader(self.colorvideo_path, start_frame=frame_ind)
                self.colors_iter = iter(self.colors_reader)
        dmap_frame = next(self.dmaps_iter)
        self.next_frame_ind = frame_ind+1
        if self.colorvideo_path is not None:
            color_frame = next(self.colors_iter)
            color_frame = np.concatenate([color_frame, np.full(color_frame.shape[:-1] + (1,), 255, np.uint8)], axis=2)
            return dmap_frame, color_frame
        else:
            return dmap_frame, None

    def get_curr_pointcloud(self):
        if self.current_frame_cloud is not None and self.current_frame_cloud[1] == self.current_sequence_frame_ind:
            return self.current_frame_cloud[0]
        depth, colors = self.load_depth_color(self.current_sequence_frame_ind)
        nanmask = depth == 0
        d = depth.copy().astype(np.float64) / 1000.
        d[nanmask] = np.nan
        pc = self.pc_table_ext * d[..., np.newaxis]
        pc_validmask = np.isfinite(pc[:, :, 0])
        pc = pc[pc_validmask]
        if colors is None or self.color_override:
            colors = np.tile(np.array(self.color).astype(np.uint8).reshape(1, 4), (len(pc), 1))
        else:
            colors = colors[pc_validmask, :]
        cloud = self.PointcloudContainer(pc, colors)
        self.current_frame_cloud = (cloud, self.current_sequence_frame_ind)
        return cloud
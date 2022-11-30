import os
import json
import pickle
import trimesh
import numpy as np
import fnmatch
from zipfile import ZipFile
from io import BytesIO
from typing import Union, Tuple, List, Sequence

def list_zip(zippath: str, folder: str = '') -> List[str]:
    input_zip = ZipFile(zippath)
    if folder == '':
        return list(input_zip.namelist())
    else:
        if folder[-1]!='/':
            folder = folder + '/'
        return list(filter(lambda x: x.startswith(folder) and x != folder, input_zip.namelist()))

def get_closest_ind_after(arr_times, current_time):
    diff = arr_times - current_time
    mask = diff > 0
    if mask.sum() == 0:
        return len(arr_times) - 1
    mask_inds = np.nonzero(mask)[0]
    mindiff_ind = mask_inds[np.argmin(diff[mask])]
    return mindiff_ind


def get_closest_ind_before(arr_times, current_time):
    diff = arr_times - current_time
    mask = diff < 0
    if mask.sum() == 0:
        return 0
    mask_inds = np.nonzero(mask)[0]
    mindiff_ind = mask_inds[np.argmax(diff[mask])]
    return mindiff_ind

class ObjectLocation:
    def __init__(self, translation, quaternion, time=None):
        self._translation = np.asarray(translation)
        self._quaternion = np.asarray(quaternion)
        if time is None:
            self._time = None
        else:
            self._time = float(time)

    def to_dict(self):
        return {"position": self._translation.tolist(),
                "quaternion": self._quaternion.tolist()}

    @property
    def translation(self) -> np.ndarray:
        return self._translation

    @property
    def position(self) -> np.ndarray:
        return self._translation

    @property
    def quaternion(self) -> np.ndarray:
        return self._quaternion

    @property
    def time(self) -> float:
        return self._time

    def __getitem__(self, item):
        if item in ["position", "translation"]:
            return self.translation
        elif item == "quaternion":
            return self.quaternion
        elif item == "time":
            return self.time
        else:
            raise IndexError(f"No such index '{item}'")

class ObjectTrajectory:
    def __init__(self, traj_poses, traj_quats, traj_times):
        assert len(traj_poses) == len(traj_quats) == len(traj_times)
        self._translations = np.asarray(traj_poses)
        self._quaternions = np.asarray(traj_quats)
        self._times = np.asarray(traj_times)

    @property
    def translations(self) -> np.ndarray:
        return self._translations

    @property
    def positions(self) -> np.ndarray:
        return self._translations

    @property
    def quaternions(self) -> np.ndarray:
        return self._quaternions

    @property
    def times(self) -> np.ndarray:
        return self._times

    def __getitem__(self, item: int):
        return ObjectLocation(self.translations[item], self.quaternions[item], time=self.times[item])

    def __len__(self):
        return len(self._translations)

    @classmethod
    def cat_trajectories(cls, traj_list: List["ObjectTrajectory"]):
        transls = []
        quats = []
        times = []
        for traj in traj_list:
            transls.append(traj.translations)
            quats.append(traj.quaternions)
            times.append(traj.times)
        res = cls(np.concatenate(transls, axis=0), np.concatenate(quats, axis=0), np.concatenate(times, axis=0))
        return res


def open_from_zip(zippath: str, datapath: str, return_zip_path: bool = False) -> Union[BytesIO, Tuple[BytesIO, str]]:
    """
    Finds and opens the file inside the zip archive
    Args:
        zippath: path to .zip archive
        datapath: path to file inside .zip, can be wildcard (i.e. 'dirname/*a??b.png').
            In case wildcard is supplied, there should be only one file matching it in the archive.
        return_zip_path: whether to return the path of the loaded file inside .zip
    Returns:
        BytesIO: filehandler containing the loaded file
        str: path to the opened file (if return_zip_path == True)
    """
    input_zip = ZipFile(zippath)
    match_fn = lambda x: fnmatch.fnmatch(x, datapath)
    filenames = list(filter(match_fn, input_zip.namelist()))
    if len(filenames) == 0:
        raise FileNotFoundError("No file matching '{}' in archive".format(datapath))
    elif len(filenames) > 1:
        raise FileNotFoundError("More than one file matching '{}' exists in archive: {}".format(datapath, filenames))
    else:
        filename = filenames[0]
        filehandler = BytesIO(input_zip.read(filename))
        if return_zip_path:
            return filehandler, filename
        return filehandler


def trimesh_load_from_zip(zippath: str, datapath: str) -> Union[trimesh.Trimesh, trimesh.points.PointCloud]:
    filehandler, filename = open_from_zip(zippath, datapath, return_zip_path = True)
    ext = os.path.splitext(filename)[1][1:]
    mesh = trimesh.load(filehandler, ext, process=False)
    return mesh


def get_camera_position(xyz_ang: Sequence[float], pos: Sequence[float]):
    camera_pose = np.array([
        [1.0, 0, 0, pos[0]],
        [0.0, 1.0, 0.0, pos[1]],
        [0.0, 0, 1.0, pos[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    sin, cos = [np.sin(a) for a in xyz_ang], [np.cos(a) for a in xyz_ang]
    x_rot = np.array(
        [
            [1.0, 0, 0, 0.0],
            [0.0, cos[0], -sin[0], 0.0],
            [0.0, sin[0], cos[0], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    y_rot = np.array(
        [
            [cos[1], 0, sin[1], 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin[1], 0, cos[1], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    z_rot = np.array(
        [
            [cos[2], -sin[2], 0, 0.0],
            [sin[2], cos[2], 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    return camera_pose.dot(z_rot.dot(y_rot.dot(x_rot)))

def load_hps_sequence(poses_pickle_path, shape_json_path):
    pkl_seq = pickle.load(open(poses_pickle_path, 'rb'))
    shape = np.array(json.load(open(shape_json_path))['betas'])
    res = [{"pose": pose, "translation": translation, "shape": shape}
           for pose, translation in zip(pkl_seq['poses'], pkl_seq['transes'])]
    return res

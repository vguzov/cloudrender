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

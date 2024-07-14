'''
Adapted from Polycam Inc.

Created by Chris Heinrich on Tuesday, 1st November 2022
Copyright © 2022 Polycam Inc. All rights reserved.
'''

"""
The CaptureFolder and associated types are the main wrappers around Polycam data as it is laid out on the filesystem. 

Initializing a CapureFolder from the absolute path of the data folder on the filesystem is usually the first step to 
working with a raw Polycam dataset.
"""

import os
import json
import numpy as np

from enum import Enum
from typing import List
import logging
from scipy.spatial.transform import Rotation as R


class CaptureArtifact(Enum):
    IMAGES = "keyframes/images"
    CORRECTED_IMAGES = "keyframes/corrected_images"
    CAMERAS = "keyframes/cameras"
    CORRECTED_CAMERAS = "keyframes/corrected_cameras"
    DEPTH_MAPS = "keyframes/depth"
    CONFIDENCE_MAPS = "keyframes/confidence"
    MESH_INFO = "mesh_info.json"
    ANCHORS = "anchors.json"
    PREVIEW_MESH = "mesh.obj"

class BBox3D:
    def __init__(self, bbox_min: np.array, bbox_max: np.array):
        """
        Initializes a 3D axis-aligned bbox from the min and max points
        
        Args:
            bbox_min: the minimum point, expected to be a 
        """
        assert bbox_min.shape == np.shape([0,0,0]), "bbox_min must be a numpy array of shape (3,)"
        assert bbox_max.shape == np.shape([0,0,0]), "bbox_max must be a numpy array of shape (3,)"
        self.min = bbox_min
        self.max = bbox_max
    
    def center(self) -> np.array:
        return (self.min + self.max) / 2.0
    
    def size(self) -> np.array:
        return (self.max - self.min)

    def __str__(self):
        return "\n *** BBox3D *** \nmin: {}\nmax: {}\nsize: {}\ncenter: {}\n".format(self.min, self.max, self.size(), self.center())


def bbox_from_points(points: List[np.array]) -> BBox3D:
    """
    Generates a bounding box from a list of points (i.e. 3D numpy arrays)
    """
    finfo = np.finfo(np.float32)
    bbox_min = np.asarray([finfo.max, finfo.max, finfo.max])
    bbox_max = np.asarray([finfo.min, finfo.min, finfo.min])
    for point in points:
        # x
        if point[0] < bbox_min[0]:
            bbox_min[0] = point[0]
        if point[0] > bbox_max[0]:
            bbox_max[0] = point[0]
        # y
        if point[1] < bbox_min[1]:
            bbox_min[1] = point[1]
        if point[1] > bbox_max[1]:
            bbox_max[1] = point[1]
         #z
        if point[2] < bbox_min[2]:
            bbox_min[2] = point[2]
        if point[2] > bbox_max[2]:
            bbox_max[2] = point[2]

    return BBox3D(bbox_min, bbox_max)

"""
Helper functions to make it faster to work with Python's standard logging library
"""

def set_log_level(logger, default_level=logging.INFO):
    if "LOG_LEVEL" in os.environ:
        level = os.environ["LOG_LEVEL"].upper()
        exec("logger.setLevel(logging.{})".format(level))
    else:
        logger.setLevel(default_level)


def setup_logger(name):
    logger = logging.getLogger(name)
    set_log_level(logger)
    logging.basicConfig()
    return logger


def class_logger(obj):
    """ initializes a logger with type name of obj """
    return setup_logger(type(obj).__name__)


def file_logger(file):
    this_file = os.path.splitext(os.path.basename(file))[0]
    logger = setup_logger(' ' + this_file + ' ')
    return logger


# A common logger instance that can be used when a dedicated
# named logger is not needed
logger = setup_logger('polyform')


class Camera:
    def __init__(self, j: dict, rotate: bool = False):
        """ Initializes a Camera object from the Polycam camera json format 
        Args:
            j: json representation of a camera object
            rotate: rotates transform data to use the instant-ngp/nerftsudio convention (default)
        """
        self.fx = j["fx"]
        self.fy = j["fy"]
        self.cx = j["cx"]
        self.cy = j["cy"]
        self.width = j["width"]
        self.height = j["height"]
        self.blur_score = j["blur_score"]
        if rotate:
            self.transform_rows = [
                [j["t_20"], j["t_21"], j["t_22"], j["t_23"]],
                [j["t_00"], j["t_01"], j["t_02"], j["t_03"]],
                [j["t_10"], j["t_11"], j["t_12"], j["t_13"]],
                [0.0,0.0,0.0,1.0]]   
        else:
            self.transform_rows = [
                [j["t_00"], j["t_01"], j["t_02"], j["t_03"]],
                [j["t_10"], j["t_11"], j["t_12"], j["t_13"]],
                [j["t_20"], j["t_21"], j["t_22"], j["t_23"]],
                [0.0,0.0,0.0,1.0]]
        self.transform = np.asarray(self.transform_rows, dtype=np.float32)

    def get_position(self):
        """ Extracts the position from the transformation matrix. """
        position_x = self.transform[0, 3]
        position_y = self.transform[1, 3]
        position_z = self.transform[2, 3]
        return position_x, position_y, position_z

    def get_quaternion(self):
        """ Extracts the rotation as a quaternion from the transformation matrix. """
        rotation_matrix = self.transform[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns (x, y, z, w) quaternion
        quaternion_x = quaternion[0]
        quaternion_y = quaternion[1]
        quaternion_z = quaternion[2]
        quaternion_w = quaternion[3]
        return quaternion_x, quaternion_y, quaternion_z, quaternion_w

    def get_pose(self):
        """ Returns the full pose as position and quaternion. """
        position = self.get_position()
        quaternion = self.get_quaternion()
        return {
            "position_x": position[0],
            "position_y": position[1],
            "position_z": position[2],
            "quaternion_x": quaternion[0],
            "quaternion_y": quaternion[1],
            "quaternion_z": quaternion[2],
            "quaternion_w": quaternion[3]
        }

class Keyframe:
    """
    A Keyframe includes the camera information (extrinsics and intrinsics) as well as the 
    path to the associated data on disk (images, depth map, confidence)
    """
    def __init__(self, folder: str, timestamp: int, rotate: bool):
        self.folder = folder
        self.timestamp = timestamp
        self.image_path = os.path.join(
            folder, "{}/{}.jpg".format(CaptureArtifact.IMAGES.value, timestamp))
        self.corrected_image_path = os.path.join(
            folder, "{}/{}.jpg".format(CaptureArtifact.CORRECTED_IMAGES.value, timestamp))
        self.camera_path = os.path.join(
            folder, "{}/{}.json".format(CaptureArtifact.CAMERAS.value, timestamp))
        self.corrected_camera_path = os.path.join(
            folder, "{}/{}.json".format(CaptureArtifact.CORRECTED_CAMERAS.value, timestamp))
        self.depth_path = os.path.join(
            folder, "{}/{}.png".format(CaptureArtifact.DEPTH_MAPS.value, timestamp))
        self.camera = Camera(self.get_best_camera_json(), rotate)

    def is_valid(self) -> bool:
        if not os.path.isfile(self.camera_path):
            return False
        if not os.path.isfile(self.image_path):
            return False
        if not os.path.isfile(self.depth_path):
            return False
        return True

    def is_optimized(self) -> bool:
        if not os.path.isfile(self.corrected_camera_path):
            return False
        if not os.path.isfile(self.corrected_image_path):
            return False
        return True

    def get_best_camera_json(self) -> dict:
        """ Returns the camera json for the optimized camera if it exists, othewise returns the ARKit camera """
        if self.is_optimized():
            return CaptureFolder.load_json(self.corrected_camera_path)
        else:
            return CaptureFolder.load_json(self.camera_path)
            
    def __str__(self):
        return "keyframe:{}".format(self.timestamp)


class CaptureFolder:
    def __init__(self, root: str):
        self.root = root
        self.id = os.path.basename(os.path.normpath(root))
        if not self.has_optimized_poses():
            logger.warning("Camera poses have not been optimized, the ARKit poses will be used as a fallback")

    def get_artifact_path(self, artifact: CaptureArtifact) -> str:
        return os.path.join(self.root, artifact.value)

    def get_artifact_paths(self, folder_artifact: CaptureArtifact,
                           file_extension: str) -> List[str]:
        """ 
        Returns all of the artifacts located in the folder_artifact with the provided file_extension
        """
        paths = []
        folder_path = self.get_artifact_path(folder_artifact)
        if not os.path.isdir(folder_path):
            return paths
        return [os.path.join(folder_path, path) for path in sorted(os.listdir(folder_path)) if path.endswith(file_extension)]

    def has_artifact(self, artifact: CaptureArtifact) -> bool:
        return os.path.exists(self.get_artifact_path(artifact))

    def has_optimized_poses(self) -> bool:
        return (self.has_artifact(CaptureArtifact.CORRECTED_CAMERAS) and self.has_artifact(CaptureArtifact.CORRECTED_IMAGES))

    def get_image_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.IMAGES, "jpg")

    def get_camera_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.CAMERAS, "json")

    def get_depth_paths(self) -> List[str]:
        return self.get_artifact_paths(CaptureArtifact.DEPTH_IMAGES, "png")

    def get_keyframe_timestamps(self) -> List[int]:
        timestamps = []
        folder_path = self.get_artifact_path(CaptureArtifact.CAMERAS)
        if not os.path.isdir(folder_path):
            return timestamps
        return [int(path.replace(".json", "")) for path in sorted(os.listdir(folder_path)) if path.endswith("json")]

    def get_keyframes(self, rotate: bool = False) -> List[Keyframe]:
        """
        Returns all valid keyframes associated with this dataset
        """
        keyframes = []
        for ts in self.get_keyframe_timestamps():
            keyframe = Keyframe(self.root, ts, rotate)
            if keyframe.is_valid():
                keyframes.append(keyframe)
        return keyframes

    @staticmethod
    def load_json(path) -> dict:
        name, ext = os.path.splitext(path)
        if not os.path.exists(path) or ext.lower() != ".json":
            print("File at path {} did not exist or was not a json file. Returning empty dict".format(path))
            return {}
        else:
            with open(path) as f:
                return json.load(f)

    def load_json_artifact(self, folder_artifact: CaptureArtifact) -> dict:
        path = self.get_artifact_path(folder_artifact)
        return CaptureFolder.load_json(path)

    @staticmethod
    def camera_bbox(keyframes: List[Keyframe]) -> BBox3D:
        """
        Returns the bounding box that contains all the cameras
        """
        positions = []
        for keyframe in keyframes:
            positions.append(keyframe.camera.transform[0:3,3])
        return bbox_from_points(points=positions)
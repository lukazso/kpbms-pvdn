from typing import List
import numpy as np
from cv2 import cv2

from pvdn import Vehicle, Instance


def flatten_vehicles(vehicles: List[Vehicle]):
    """
    Compresses the hierarchical structure of the vehicles list from the PVDN dataset to a simple list with all
    keypoints. This removes the information about the correspondence between KP and vehicle.
    :param vehicles: a list of vehicles
    :return: List[Instance]
    """
    kps: list[Instance] = []
    for v in vehicles:
        kps.extend(v.instances)

    return kps


def draw_keypoints(kps: List[Vehicle], img: np.ndarray):
    """
    Draws keypoints from vehicle instances into an images.
    :param vehicles: List of vehicle instances
    :param img: image as a numpy array. Note that it has to be 3 channels! Otherwise
        color cannot be seen.
    """

    for kp in kps:
        if kp.direct:
            draw_direct_kp(kp.position, img)
        else:
            draw_indirect_kp(kp.position, img)

    return img


def draw_direct_kp(kp: List[int], img: np.ndarray):
    cv2.circle(img, center=tuple(kp), thickness=-1, radius=3, color=(255, 0, 0))


def draw_indirect_kp(kp: List[int], img: np.ndarray):
    cv2.circle(img, center=tuple(kp), thickness=-1, radius=3, color=(0, 255, 0))


def kp_in_box(kp: List, box: List):
    """
    Checks if a keypoint lies within a bounding box.
    """
    kx, ky = kp
    bx1, by1, bx2, by2 = box
    return (bx1 <= kx <= bx2) and (by1 <= ky <= by2)
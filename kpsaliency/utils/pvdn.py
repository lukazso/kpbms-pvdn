from typing import List, Tuple
import numpy as np
import cv2
from copy import deepcopy
from enum import Enum

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


def resize(img: np.array, vehicles: List[Vehicle], img_size: List[int],
           interpolation: Enum) -> Tuple[np.array, List[Vehicle]]:
    """
    Resizes image with corresponding keypoint annotations
    :param img: np.uint8 array of shape [h, w, c] or [h, w]
    :param vehicles: list of Vehicle objects representing keypoint annotations
    :param img_size: new img size as [h, w]
    :param interpolation: cv2 interpolation mode (e.g., cv2.INTER_LINEAR)
    :return:
        img: resized image as a copy of the original image
        vehicles: resized annotations as a copy of the original annotations
    """

    img = img.copy()
    horig, worig = img.shape[:2]
    hnew, wnew = img_size
    hscale = hnew / horig
    wscale = wnew / worig

    img = cv2.resize(img, (wnew, hnew), interpolation)

    vehicles = deepcopy(vehicles)
    for vehicle in vehicles:
        px, py = vehicle.position
        px = int(px * wscale)
        py = int(py * hscale)
        vehicle.position = (px, py)
        for inst in vehicle.instances:
            px, py = inst.position
            px = int(px * wscale)
            py = int(py * hscale)
            inst.position = (px, py)

    return img, vehicles

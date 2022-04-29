import random
from typing import List
import os
import json

import cv2.cv2
from pvdn import Instance

from kpsaliency.generators.generator import SaliencyGenerator

from kpsaliency.utils import clamp, determine_regional_val
from kpsaliency.utils.misc import make_gaussian

import numpy as np
from skimage.morphology import disk, flood


FILE_DIR = os.path.dirname(__file__)


class KPBMSGenerator(SaliencyGenerator):
    INIT_KERNEL_SIZE = 20

    def __init__(self, lower_direct: float = 0.9, upper_direct: float = 1.0,
                 lower_indirect: float = 0.8,
                 upper_indirect: float = 1.2, n: int = 10,
                 selem: np.array = disk(5), sigma: float = 0.0,
                 kernel_size: int = 300):
        """
        :type selem: Structural element used to calculate a more stable intensity value around kp
        :param lower_direct: lower relative boundary for direct instances
        :param upper_direct: upper relative boundary for direct instances
        :param lower_indirect: lower relative boundary for indirect instances
        :param upper_indirect: upper relative boundary for indirect instances
        :param n: number of thresholds used in BMS
        :param sigma: sigma of gauss kernel used to blurr the image before the BMS algorithm
        """
        # check that params make sense
        assert lower_direct > 0
        assert lower_indirect > 0
        assert upper_indirect > lower_indirect
        assert upper_direct > lower_direct

        self.lower_direct = lower_direct
        self.upper_direct = upper_direct
        self.lower_indirect = lower_indirect
        self.upper_indirect = upper_indirect
        self.n = n
        self.selem = selem
        self.sigma = sigma
        self.kernel_size = kernel_size
        if kernel_size:
            self.kh, self.kw = kernel_size, kernel_size

            self.kernel = make_gaussian(self.INIT_KERNEL_SIZE,
                                        fwhm=self.INIT_KERNEL_SIZE // 2 - 1)
            self.kernel = cv2.resize(self.kernel, dsize=(self.kh, self.kw))

    @property
    def params(self):
        return {
            "lower_direct": self.lower_direct,
            "upper_direct": self.upper_direct,
            "lower_indirect": self.lower_indirect,
            "upper_indirect": self.upper_indirect,
            "n": self.n,
            "sigma": self.sigma,
            "selem_size": self.selem.shape[0],
            "kernel_size": self.kernel_size
        }

    @staticmethod
    def from_json(config_path: str = None):
        """
        Instanciates the KPMSGenerator from a json config.
        :param config_path: /path/to/config.json
        :return: instance of KPBMSGenerator
        """
        # if no file is provided, use the default config
        if not config_path:
            config_path = os.path.join(FILE_DIR, "default_params.json")

        # error catching
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"{config_path} is either not a file or does "
                                    f"not exist.")

        # read config from json
        with open(config_path, "r") as f:
            params = json.load(f)

        selem = disk(params["selem_size"])

        return KPBMSGenerator(
            upper_direct=params["upper_direct"], lower_direct=params["lower_direct"],
            upper_indirect=params["upper_indirect"],
            lower_indirect=params["lower_indirect"],
            n=params["n"], sigma=params["sigma"], selem=selem,
            kernel_size=params["kernel_size"]
        )

    @staticmethod
    def from_dict(params: dict):
        selem = disk(int(params["selem_size"]))
        return KPBMSGenerator(
            upper_direct=params["upper_direct"], lower_direct=params["lower_direct"],
            upper_indirect=params["upper_indirect"],
            lower_indirect=params["lower_indirect"],
            n=int(params["n"]), sigma=params["sigma"], selem=selem,
            kernel_size=int(params["kernel_size"])
        )

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.params, f)

    def compute_bmap_for_kp(self, inst, img):
        h, w = img.shape[:2]

        ix, iy = inst.position
        direct = inst.direct

        # ==== 1. step: specify search range and values based on kp ====
        # sample intensity value around position of instance and make sure that value is within [1,254]
        val, coord = determine_regional_val(self.selem, img.copy(), ix, iy, method="max")

        # update ix, iy to the new coordinates where the regional value was determined
        ix, iy = coord

        # calculate bounds around this intensity
        lower_bound = int(
            val * self.lower_direct if direct else val * self.lower_indirect)
        upper_bound = int(
            val * self.upper_direct if direct else val * self.upper_indirect)

        # clamp bound so they stay within [0,255]
        lower_bound = clamp(0, val - 1, lower_bound)
        upper_bound = clamp(val + 1, 255, upper_bound)

        # specify threshold values
        # theta = np.arange(lower_bound, upper_bound, self.n)
        theta = np.linspace(lower_bound, upper_bound, self.n)

        # ==== 2. step create boolean maps ====
        b_maps = np.zeros(shape=(theta.shape[0], h, w), dtype=bool)

        # ==== 3. step make gaussian window around keypoint to avoid huge saliency
        # blobs ====
        if not self.kernel_size:
            _img = img
        else:
            kernel_mask = np.zeros(shape=(h, w))

            kw_low = self.kw // 2 + min(0, ix - self.kw // 2)
            kw_high = self.kw // 2 + min(0, w - (ix + self.kw // 2))

            kh_low = self.kh // 2 + min(0, iy - self.kh // 2)
            kh_high = self.kh // 2 + min(0, h - (iy + self.kh // 2))

            _kernel = self.kernel[self.kh // 2 - kh_low: self.kh // 2 + kh_high,
                      self.kw // 2 - kw_low:self.kw // 2 + kw_high]

            cv2.normalize(_kernel, dst=_kernel, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
            kernel_mask[iy - kh_low:iy + kh_high, ix - kw_low:ix + kw_high] += _kernel
            _img = (img.copy() * kernel_mask).astype(np.uint8)

        # create boolean masks
        for i, t in enumerate(theta):
            # calculate binary mask for given threshold
            b_map = np.zeros(shape=(img.shape[0], img.shape[1]))

            b_map[:, :] = _img > np.ones_like(_img) * t

            # flood fill algorithm from KP
            mask = (flood(b_map, (iy, ix)))
            b_map = np.logical_and(b_map, mask)

            # update overall b map
            b_maps[i] = b_map

        kp_s_map = np.mean(b_maps, axis=0)

        return kp_s_map

    def generate_single_bm(self, image: np.array, kps: List[Instance]) -> List[
        np.ndarray]:
        """
        Generates a saliency map for each instance.
        :param image: input image as a numpy array of type int (0-255)
        :param kps: list of keypoint instances for the input image
        :return: list of saliency maps, where the i-th entry in the saliency map list
            corresponds to the i-th entry in the keypoint list. If the keypoint list
            is empty then also an empty list is returned.
        """
        # setup data
        h, w = image.shape[:2]

        # blurr image
        img = cv2.cv2.GaussianBlur(image, (21, 5), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        s_maps = []
        for inst in kps:
            s_map = self.compute_bmap_for_kp(inst, img)
            s_maps.append(s_map)

        return s_maps

    def generate_cumulated_sm(self, image: np.array, kps: List[Instance]) -> \
            np.ndarray:
        """
        Combines all single keypoint specific saliency map to each a saliency map.
        :param image: input image as a numpy array of type int (0-255) of shape [h, w]
        :param kps: list of keypoint instances for the input image
        :return: single overlayed saliency map of all keypoints
        """
        s_maps = self.generate_single_bm(image, kps)

        if len(s_maps) == 0:
            return np.zeros_like(image)

        # cumulate maps for final sm
        s_map = np.zeros_like(s_maps[0])
        for kp_s_map in s_maps:
            s_map += kp_s_map

        # normalize sm to be in [0,1]
        with np.errstate(invalid='ignore'):
            s_map /= np.max(s_map)

        return s_map

    def __repr__(self):
        return f"KPBMS \n direct:[{self.lower_direct}, {self.upper_direct}, " \
               f"\n indirect:[{self.lower_indirect}, {self.upper_indirect}], " \
               f"\n n:{self.n}, σ:{self.sigma} "

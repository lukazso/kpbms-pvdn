from typing import List

from pvdn.core import PVDNDataset
from pvdn.keypoints import Instance
import numpy as np
from kpsaliency.utils.pvdn import flatten_vehicles


class SaliencyGenerator:
    def generate_sm(self, img: np.array, kps: List[Instance]):
        return np.zeros_like(img)

    def run_on_set(self, data: PVDNDataset):
        sm_list = []

        # run over all frames in dataset
        for info, img, vehicles in data:
            # collect all kps
            instances = flatten_vehicles(vehicles)

            # generate sm
            sm_list.append(self.generate_sm(img, instances))

        return sm_list

    def __repr__(self):
        return "abstr. Saliency"

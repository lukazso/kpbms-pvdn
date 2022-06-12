import argparse
import json
from typing import List, Any, Dict, Union, Tuple
import os
from enum import Enum

import multiprocessing as mp
import numpy as np

from pvdn import PVDNDataset, Instance

from kpsaliency.generators.bboxes import KeypointBoxGenerator, KPBMSBoxGenerator, \
    KPBMSGenerator
from kpsaliency.utils import flatten_vehicles, kp_in_box
from kpsaliency.utils.misc import handle_pbar


class Labels(Enum):
    DIRECT = 1
    INDIRECT = 2


class SaliencyBoxDataset(PVDNDataset):
    def __init__(self, path: str, filters: List[Any] = [], transform=None,
                 read_annots: bool = True, load_images: bool = True,
                 keypoints_path: str = None, bbox_path = None):
        super(SaliencyBoxDataset, self).__init__(path, filters, None, read_annots,
                                              load_images, keypoints_path)
        self.bbox_path = bbox_path if bbox_path else os.path.join(self.labels_path,
                                                                  "kpbms_boxes")

    def __getitem__(self, idx):
        """
        :return:
            img: image as numpy array or torch tensor
            info: pvdn.meta.ImageInformation
            vehicle: pvdn.core.Vehicle
            bboxes: list of n bboxes, where each bbox is [x1, y1, x2, y2]
            labels: list of n labels (enum Labels) corresponding to bboxes
        """
        img, info, vehicles = super().__getitem__(idx)

        annotation_path = os.path.join(self.bbox_path, info.file_name.split(".")[0]
                                       + ".json")
        with open(annotation_path, "r") as f:
            annot = json.load(f)

        bboxes = annot["bounding_boxes"]
        labels = annot["labels"]

        return img, info, vehicles, bboxes, labels

    def generate_dataset(self, box_generator: Union[KPBMSBoxGenerator,
                                                    Dict[str, KPBMSBoxGenerator]],
                         verbose: bool = False, n_workers: int = 4) -> None:
        """
        Generates the bounding box dataset based on the saliency maps (uses
            multiprocessing).
        :param box_generator:
        :param verbose:
        :param n_workers:
        :return:
        """
        print(f"Generating {self.bbox_path}")
        os.makedirs(self.bbox_path, exist_ok=True)

        total = len(self)
        batch_size = total // n_workers
        batch_idxs = [batch_size * n for n in range(n_workers)]
        batches = [self.img_idx[n:n + batch_size] if n != batch_idxs[-1]
                   else self.img_idx[n:] for n in batch_idxs]

        pbar_queue = mp.Queue()
        desc = "Generating saliency bounding box dataset"
        if verbose:
            pbar_proc = mp.Process(target=handle_pbar, args=(pbar_queue, total, desc))
            pbar_proc.start()

        processes = [mp.Process(target=self._generate_dataset_batch,
                                args=(box_generator, batch, pbar_queue, offset))
                     for batch, offset in zip(batches, batch_idxs)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        if verbose:
            pbar_proc.join()

    def _generate_dataset_batch(self, generator: Union[KPBMSBoxGenerator,
                                                       Dict[str, KPBMSBoxGenerator]],
                                img_idxs: List[int], pbar_queue: mp.Queue,
                                offset: int) -> None:
        _generator = generator
        for idx, id in enumerate(img_idxs):
            img, info, vehicles = super().__getitem__(idx + offset)
            if type(generator) is dict:
                _generator = generator[info.sequence.directory]
            kps = flatten_vehicles(vehicles)

            bounding_boxes, labels = self.generate_annotations(img, kps, _generator)

            # save annotation
            annotation_path = os.path.join(self.bbox_path, "{:06d}.json".format(id))
            self.save_annotations(bounding_boxes, labels, annotation_path)

            if pbar_queue:
                # update progress bar
                pbar_queue.put(1)

    @staticmethod
    def generate_annotations(img: np.ndarray, kps: List[Instance],
                             generator: KPBMSBoxGenerator) \
            -> Tuple[List[Union[Tuple, List]], List[int]]:
        """
        :param img: grayscale img as np.uint8 array [h, w]
        :param kps: list of pvdn.Instance objects
        :param generator: KPBMSBoxGenerator object
        :return:
            bboxes: list of n bboxes, where each bbox is [x1, y1, x2, y2]
            labels: list of n integer labels (each label either 1 or 2)
                corresponding to bboxes
        """
        bboxes = generator.propose(img, kps)
        labels = []
        if bboxes:
            labels = SaliencyBoxDataset.label_bboxes(kps, bboxes)
            labels = [label.value for label in labels]

        return bboxes, labels

    @staticmethod
    def save_annotations(bboxes: List[Union[Tuple, List]], labels: List[Instance],
                         path: str) -> None:
        """
        :param bboxes: list of n bboxes, where each bbox is [x1, y1, x2, y2]
        :param labels: list of n labels (enum Labels) corresponding to bboxes
        :param path: full .json file path to write the annotations to
        """
        with open(path, "w") as f:
            annotation = {"bounding_boxes": bboxes,
                          "labels": labels}
            json.dump(annotation, f)

    @staticmethod
    def label_bboxes(kps, bboxes) -> List[Labels]:
        """
        :param kps: list of pvdn.Instance objects
        :param bboxes: list of n bboxes, where each bbox is [x1, y1, x2, y2]
        :return: list of n labels (enum Labels) corresponding to bboxes
        """
        labels = []
        for box in bboxes:
            label = []
            # check if a keypoint lies within the box
            for kp in kps:
                if kp_in_box(list(kp.position), box):
                    if kp.direct:
                        label.append(Labels.DIRECT)
                    else:
                        label.append(Labels.INDIRECT)

            # check if all labels are equal
            label = list(set(label))
            if len(label) == 1:
                label = label[0]
            else:
                label = Labels.DIRECT

            labels.append(label)

        return labels


if __name__ == "__main__":
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", "-d", type=str,
                        default="/home/lukas/Development/datasets/PVDN/day")
    opts = parser.parse_args()

    # for split in ("train", "val", "test"):
    split = "train"
    print("Split:", split)

    base_path = os.path.join(opts.data_path, split)

    bbox_path = os.path.join(base_path, "labels/kpbms_boxes")
    dataset = SaliencyBoxDataset(path=base_path,
                              bbox_path=bbox_path)
    pvdn_dataset = PVDNDataset(base_path)

    params_dir = os.path.join(base_path, "labels/kpbms_params")
    params_paths = [os.path.join(params_dir, file) for file in os.listdir(params_dir)]
    generators = {}
    for file in params_paths:
        scene = file.split("/")[-1].split(".")[0]
        generators[scene] = KPBMSBoxGenerator(KPBMSGenerator.from_json(file))

    # dataset.generate_dataset(generator=generators, n_workers=4, verbose=True)
    g = list(generators.values())[0]
    g = generators["S00285"]
    # pbar = tqdm(pvdn_dataset)
    # for img, info, vehicles in pbar:
    #     pbar.desc = info.file_name
    #     pbar.update()
        # print(info.file_name)
    i = np.where(np.array(pvdn_dataset.img_idx) == 98412)[0].item()
    img, info, vehicles = pvdn_dataset[i]
    print(info.file_name)
    kps = flatten_vehicles(vehicles)
    dataset.generate_annotations(img, kps, g)

    del dataset

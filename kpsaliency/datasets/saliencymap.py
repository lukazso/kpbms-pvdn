# common libraries
import json
import os
import time
from typing import List, Any, Union, Dict
from tqdm import tqdm

import numpy as np
import cv2
from skimage.morphology import disk
import multiprocessing as mp

# pvdn dependencies
from pvdn import PVDNDataset

# own library dependencies
from kpsaliency.generators import KPBMSGenerator
from kpsaliency.utils.misc import handle_pbar


def collate_fn(data):
    """
    Custom collate_fn function for DataLoader objects working with the SaliencyMapDataset class.
    :returns:
        imgs - np.ndarray of shape [batch_size, height, width, channels]    containing the input images.
        smaps_direct - np.ndarray of shape [batch_size, height, width] containing the direct saliency maps.
        smaps_indirect - np.ndarray of shape [batch_size, height, width] containing the indirect saliency maps.
        infos - list of batch_size items of types pvdn.ImageInformation containing information of each image.
        vehicles - list of batch_size items of types pvdn.Vehicle containing information about the keypoint annotations for each image.
    """
    imgs, smaps_direct, smaps_indirect, infos, vehicles = zip(*data)
    imgs = np.stack(imgs, axis=0)
    smaps_indirect = np.stack(smaps_indirect, axis=0)
    smaps_direct = np.stack(smaps_direct, axis=0)
    
    return imgs, smaps_direct, smaps_indirect, infos, vehicles


class SaliencyMapDataset(PVDNDataset):
    def __init__(self, path: str,
                 filters: List[Any] = [], transform: List[Any] = None,
                 read_annots: bool = True, load_images: bool = True,
                 keypoints_path: str = None
                 ):
        super().__init__(path, filters, transform, read_annots, load_images=load_images,
                         keypoints_path=keypoints_path)

        self.smap_path = os.path.join(self.base_path, "saliency_maps")
        self.smap_path_direct = os.path.join(self.smap_path, "direct")
        self.smap_path_indirect = os.path.join(self.smap_path, "indirect")

        self.dataset_exists = True

        if not os.path.isdir(self.smap_path):
            tqdm.write("Note that you need to generate your dataset first.")
            self.dataset_exists = False

    def generate_dataset(self, bms_generator: Union[KPBMSGenerator,
                                                    Dict[str, KPBMSGenerator]],
                         n_workers: int= 4):

        _load_images = self.load_images
        self.load_images = True
        
        # setup necessary directories
        os.makedirs(self.smap_path, exist_ok=True)
        direct_path = self.smap_path_direct
        indirect_path = self.smap_path_indirect
        os.makedirs(direct_path, exist_ok=True)
        os.makedirs(indirect_path, exist_ok=True)

        for scene in self.sequences:
            os.makedirs(os.path.join(self.smap_path_direct, scene.directory),
                        exist_ok=True)
            os.makedirs(os.path.join(self.smap_path_indirect, scene.directory),
                        exist_ok=True)

        params_path = os.path.join(self.smap_path, "params.json")
        # with open(params_path, "w") as f:
        #     json.dump(bms_generator.params, f)
        # print(f"Written parameters to {params_path}.")
        print(f"Running with {n_workers} workers.")
        total = len(self)
        batch_size = total // n_workers
        batch_idxs = [batch_size * n for n in range(n_workers)]
        idxs = list(range(total))
        batches = [idxs[n:n+batch_size] if n != batch_idxs[-1] else idxs[n:]
                   for n in batch_idxs]
        pbar_queue = mp.Queue()

        desc = "Generating saliency map dataset"
        pbar_proc = mp.Process(target=handle_pbar, args=(pbar_queue, total, desc))
        pbar_proc.start()

        processes = [mp.Process(target=SaliencyMapDataset._generate_dataset_batch,
                                args=(self, batch, bms_generator, pbar_queue))
                     for batch in batches]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        pbar_proc.join()

        self.load_images = _load_images
        self.dataset_exists = True

    def _generate_dataset_batch(self, idxs, bms_generator, pbar_queue: mp.Queue):
        _bms_generator = bms_generator
        for i in idxs:
            img, info, vehicles = super().__getitem__(i)

            if type(bms_generator) is dict:
                _bms_generator = bms_generator[info.sequence.directory]

            direct_kps = []
            indirect_kps = []

            for vehicle in vehicles:
                for instance in vehicle.instances:
                    if instance.direct:
                        direct_kps.append(instance)
                    else:
                        indirect_kps.append(instance)

            direct_smaps = _bms_generator.generate_cumulated_sm(img, direct_kps)

            indirect_smaps = _bms_generator.generate_cumulated_sm(img, indirect_kps)

            # convert maps to uint8 [0, 255]
            direct_smaps = (direct_smaps * 255).astype(np.uint8)
            indirect_smaps = (indirect_smaps * 255).astype(np.uint8)

            # save maps
            cv2.imwrite(
                os.path.join(self.smap_path_direct, info.sequence.directory,
                             info.file_name), direct_smaps)
            cv2.imwrite(
                os.path.join(self.smap_path_indirect, info.sequence.directory,
                             info.file_name), indirect_smaps)

            pbar_queue.put(1)

    def __getitem__(self, idx):
        img, info, vehicles = super().__getitem__(idx)
        smap_direct = cv2.imread(os.path.join(
            self.smap_path_direct, info.sequence.directory, info.file_name
        ), 0)
        smap_indirect = cv2.imread(os.path.join(
            self.smap_path_indirect, info.sequence.directory, info.file_name
        ), 0)

        return img, smap_direct, smap_indirect, info, vehicles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path", type=str,
                        default="/raid/datasets/PVDN_Dataset/PVDN/day")
    opts = parser.parse_args()

    for split in ("test", "val", "train"):
        tqdm.write(f"Split: {split}")
        path = os.path.join(opts.data_path, split)
        generator = KPBMSGenerator.from_json()

        params_dir = os.path.join(path, "labels/kpbms_params")
        params_paths = [os.path.join(params_dir, file) for file in os.listdir(params_dir)]

        generators = {}
        for file in params_paths:
            scene = file.split("/")[-1].split(".")[0]
            generators[scene] = KPBMSGenerator.from_json(file)

        dataset = SaliencyMapDataset(
            path=path
        )
        dataset.generate_dataset(bms_generator=generators, n_workers=7)
        del dataset

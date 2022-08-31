from pvdn import Instance
from pvdn.metrics import BoundingBoxEvaluator
from pvdn.metrics.convert import coco_to_results_format

from kpsaliency.datasets import SaliencyBoxDataset
from kpsaliency.utils import flatten_vehicles
from kpsaliency.metrics.core import precision, recall, fscore

import numpy as np

from typing import List, Dict
from tqdm import tqdm


class BasicBoxEvaluator:
    def __init__(self):
        pass

    def evaluate(self, bboxes: Dict[int, Dict[str, List]],
                 kps: List[List[int]], conf_thresh: float = 0.5):
        kp_quality_hist = []
        box_quality_hist = []

        total_scores = {"tps": 0, "fps": 0, "fns": 0}

        for img_id, kp_coords in kps.items():
            if not img_id in bboxes.keys():
                scores = np.empty(shape=[0, 1])
                bbox_coords = np.empty(shape=[0, 4])
            else:
                scores = np.array(bboxes[img_id]["scores"]).reshape(-1, 1)
                bbox_coords = np.array(bboxes[img_id]["bounding_boxes"]).reshape(-1, 4)

            bbox_coords = np.delete(bbox_coords, np.where(scores < conf_thresh),
                                    axis=0)

            kp_coords = np.array(kp_coords).reshape(    -1, 2)

            img_scores, kp_quality, box_quality = \
                BoundingBoxEvaluator.evaluate_single_image(kp_coords, bbox_coords)

            total_scores = {k: v + img_scores[k] for k, v in total_scores.items()}
            kp_quality_hist += kp_quality
            box_quality_hist += box_quality

        tp = total_scores["tps"]
        fp = total_scores["fps"]
        fn = total_scores["fns"]

        prec, rec, fsc, kp_quality, box_quality, combined = \
            BasicBoxEvaluator.calculate_metrics(tp, fp, fn, kp_quality_hist,
                                                box_quality_hist)

        box_quality_std = np.std(box_quality_hist)
        kp_quality_std = np.std(kp_quality_hist)

        return prec, rec, fsc, kp_quality, kp_quality_std, box_quality, \
               box_quality_std, combined


    @staticmethod
    def calculate_metrics(tp: int, fp: int, fn: int, kp_quality_hist,
                          box_quality_hist):
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        fsc = fscore(tp, fp, fn)

        kp_quality = np.mean(kp_quality_hist)
        box_quality = np.mean(box_quality_hist)

        combined = kp_quality * box_quality

        return prec, rec, fsc, kp_quality, box_quality, combined


class DatasetEvaluator(BasicBoxEvaluator):
    def __init__(self):
        super(DatasetEvaluator, self).__init__()

    def evaluate_dataset(self, dataset: SaliencyBoxDataset, verbose: bool = False):
        kp_dict = {}
        bbox_dict = {}

        desc = "Collecting dataset information"
        for img, info, vehicles, bboxes, labels in tqdm(dataset, desc=desc,
                                                        disable=not verbose):
            instances = flatten_vehicles(vehicles)
            kps = [list(inst.position) for inst in instances]
            img_id = info.id
            kp_dict[img_id] = kps

            scores = [1.0] * len(bboxes)
            bbox_dict[img_id] = {
                "scores": scores,
                "bounding_boxes": bboxes
            }

        if verbose:
            print("Calculating metrics...")
        prec, rec, fsc, kp_quality, kp_quality_std, box_quality, box_quality_std, \
            combined = super().evaluate(bbox_dict, kp_dict)

        if verbose:
            ndig = 2
            print(f"Precision:\t{round(prec, ndig)}")
            print(f"Recall:\t{round(rec, ndig)}")
            print(f"F-Score:\t{round(fsc, ndig)}")
            print(f"KP Quality:\t{round(kp_quality, ndig)} +- "
                  f"{round(kp_quality_std, ndig)}")
            print(f"Box Quality:\t{round(box_quality, ndig)} +- "
                  f"{round(box_quality_std, ndig)}")
            print(f"Combined:\t{round(combined, ndig)}")

        return prec, rec, fsc, kp_quality, kp_quality_std, box_quality, box_quality_std, combined

from itertools import combinations
from typing import List
from scipy.ndimage import label, find_objects
import numpy as np

from pvdn.keypoints import Instance
from pvdn.metrics import BoundingBoxEvaluator

from kpsaliency.generators import KPBMSGenerator
from kpsaliency.utils.misc import ioa


def slices_to_bbox(slices):
    y1 = slices[0].start
    y2 = slices[0].stop
    x1 = slices[1].start
    x2 = slices[1].stop
    return [x1, y1, x2, y2]


class KeypointBoxGenerator:
    def __init__(self, generator):
        self.generator = generator
    
    def propose(self, img: np.ndarray, kps: List[Instance]) -> List:
        raise NotImplementedError
    

class KPBMSBoxGenerator(KeypointBoxGenerator):
    def __init__(self, generator: KPBMSGenerator):
        super(KPBMSBoxGenerator, self).__init__(generator)

    def propose(self, img: np.ndarray, kps: List[Instance]):
        structure = np.ones((3, 3))
        bboxes = []

        # generate saliency maps for each keypoint
        smaps = self.generator.generate_single_bm(image=img, kps=kps)

        for smap in smaps:
            # find blob in each saliency map
            bboxes += self.detect_blobs(smap, structure)

        # filter the bounding boxes based on which combination gives best metric
        kp_coords = [list(inst.position) for inst in kps]

        bboxes_filterd = self.filter_proposals(bboxes, kp_coords)

        return bboxes_filterd

    @staticmethod
    def detect_blobs(smap: List[np.ndarray], structure: np.ndarray = None):
        if structure is None:
            structure = np.ones((3, 3))

        bboxes = []
        labeled = label(smap > 0, structure=structure)[0]
        slices = find_objects(labeled)
        for slice in slices:
            bbox = slices_to_bbox(slice)
            bboxes.append(bbox)

        return bboxes

    @staticmethod
    def filter_proposals(bboxes: List, kps: List):
        if not bboxes:
            return []
        if len(bboxes) == 1:
            return bboxes
        kps = np.array(kps)
        combos = []
        qualities = []
        fscores = []
        metrics = []
        for i in range(1, len(bboxes) + 1):
            cs = list(combinations(bboxes, r=i))
            cs = np.array(cs)
            for c in cs:
                faulty_combo = False
                c = c.reshape(-1, 4)
                for n in range(c.shape[0]):
                    for m in range(c.shape[0]):
                        if m != n:
                            gt = c[n].tolist()
                            pred = c[m].tolist()
                            inter = ioa(gt, pred)
                            if inter > 0.2:
                                faulty_combo = True
                                break
                    if faulty_combo:
                        break

                if not faulty_combo:
                    scores, kp_quality, box_quality = \
                        BoundingBoxEvaluator.evaluate_single_image(kps, c)
                    if box_quality and kp_quality:
                        quality = np.mean(kp_quality) * np.mean(box_quality)
                    else:
                        quality = 0
                    fscore = scores["tps"] / (scores["tps"] + scores["fns"])
                    fscores.append(fscore)
                    qualities.append(quality)
                    metric = quality * fscore
                    metrics.append(metric)
                    n = c.shape[0]
                    if n == 1:
                        combos.append(c.tolist())
                    else:
                        combos.append(c.squeeze().tolist())
                    # combos.append(c.tolist())

        max_fscore = np.max(fscores)
        max_idxs = np.where(np.array(fscores) == max_fscore)[0].astype(int).tolist()

        combos = [combos[int(i)] for i in max_idxs]
        metrics = [metrics[int(i)] for i in max_idxs]

        idxs = list(range(len(metrics)))
        metrics, idxs = zip(*sorted(zip(metrics, idxs), reverse=True))

        winning_combo = combos[idxs[0]]
        if type(winning_combo[0]) is not list:
            winning_combo = [winning_combo]
        return winning_combo

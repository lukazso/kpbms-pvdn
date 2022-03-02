import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time
from typing import List


def make_gaussian(size: int, fwhm: int = 3, center=None) -> np.ndarray:
    """ Make a square gaussian kernel.
    :param size: length of a side of the square
    :param fwhm: full-width-half-maximum, which can be thought of as an effective
        radius.
    :return: 2-dim numpy array of shape [size, size] and dtype float.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def handle_pbar(queue: mp.Queue, total: int, description: str = ""):
    pbar = tqdm(desc=description, total=total)
    while True:
        time.sleep(1)
        if pbar.total == pbar.n:
            break
        while not queue.empty():
            inc = queue.get()
            pbar.update(inc)
    pbar.close()


def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def ioa(gt: List[int], pred: List[int]):
    """ intersection over area
    :param gt: ground truth bounding box as [x1, y1, x2, y2]
    :param pred: predicted bounding box as [x1, y1, x2, y2]
    :return: intersection area / ground truth area
    """
    assert gt[0] < gt[2]
    assert gt[1] < gt[3]
    assert pred[0] < pred[2]
    assert pred[1] < pred[3]

    xa = max(gt[0], pred[0])
    ya = max(gt[1], pred[1])
    xb = min(gt[2], pred[2])
    yb = min(gt[3], pred[3])

    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    result = inter_area / gt_area
    return result

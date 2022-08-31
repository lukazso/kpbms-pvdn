

def precision(tp: int, fp: int):
    denom = tp + fp
    if denom <= 0:
        return -1
    return tp / denom


def recall(tp: int, fn: int):
    denom = tp + fn
    if denom <= 0:
        return -1
    return tp / denom


def fscore(tp: int, fp: int, fn: int):
    denom = tp + 0.5 * (fp + fn)
    if denom <= 0:
        return -1
    return tp / denom


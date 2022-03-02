import numpy as np


def clamp(l: int, h: int, v: int):
    return min(h, max(l, v))


def determine_regional_val(selem: np.array, img: np.array, x: int, y: int,
                           method: str = "max"):
    """ supporded methods are min, max, mean """
    # calculate window
    H, W = img.shape[-2:]
    h, w = selem.shape
    i_min = clamp(0, W - 1, x - w // 2)
    j_min = clamp(0, H, y - h // 2)
    i_max = clamp(0, W, x + w // 2 + 1)
    j_max = clamp(0, H, y + h // 2 + 1)

    # get image region
    reg = img[j_min:j_max, i_min:i_max]
    selem = selem[:reg.shape[0], :reg.shape[1]]

    # mask region with selem
    reg *= selem

    if method == "min":
        return np.min(reg)
    elif method == "max":
        return np.max(reg)
    elif method == "mean":
        return reg.sum() / selem.sum()
    else:
        raise KeyError(f"{method} is not supported. Supported methods are 'min', "
                       f"'max' and 'mean'.")

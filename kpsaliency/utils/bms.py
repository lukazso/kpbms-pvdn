import numpy as np


def clamp(l: int, h: int, v: int):
    return min(h, max(l, v))


def determine_regional_val(selem: np.array, img: np.array, x: int, y: int,
                           method: str = "max"):
    """ 
    supporded methods are min, max, mean 
    :return: returns value and coordinate (x, y)
    """
    # calculate window
    H, W = img.shape[-2:]
    h, w = selem.shape
    i_min = clamp(0, W - 1, x - w // 2)
    j_min = clamp(0, H, y - h // 2)
    i_max = clamp(0, W, x + w // 2 + 1)
    j_max = clamp(0, H, y + h // 2 + 1)
    h = j_max - j_min
    w = i_max - i_min
    # get image region
    reg = img[j_min:j_max, i_min:i_max]
    selem = selem[:reg.shape[0], :reg.shape[1]]

    # mask region with selem
    reg *= selem

    v, coord = None, None
    if method == "min":
        v = np.min(reg)
        coord = np.array(np.unravel_index(np.argmin(reg), shape=(h, w)))
        coord += [j_min, i_min]
        coord = coord[::-1]

    elif method == "max":
        v = np.max(reg)
        coord = np.array(np.unravel_index(np.argmax(reg), shape=(h, w)))
        coord += [j_min, i_min]
        coord = coord[::-1]

    elif method == "mean":
        v = reg.sum() / selem.sum()
        coord = [x, y]
    else:
        raise KeyError(f"{method} is not supported. Supported methods are 'min', "
                       f"'max' and 'mean'.")

    return v, coord


def draw_bms_overlay(smap: np.array, img: np.array, direct=True):
    """

    :param smap: np.uint8 array [h, c]
    :param img: np.uint8 array [h, w, 3] (note that it has to be 3 channel!!)
    :param direct: bool whether its a direct smap (this decides the color of the
        overlay)
    :return: overlayed img
    """
    if direct:
        c = 2   # red channel
    else:
        c = 1   # green channel

    smap = smap.copy()
    mask = smap.astype(bool)
    mask = np.bitwise_not(mask)

    overlay = img.copy()
    overlay[:, :, c] *= mask
    overlay[:, :, c] += smap

    return overlay

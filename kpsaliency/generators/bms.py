import json

import cv2
from skimage.morphology import disk, flood
import numpy as np


def clamp(l: int, h: int, v: int):
    return min(h, max(l, v))


def compute_bm(l,h,t,img):    
    threshold = clamp(l,h,t)
    bm = np.zeros(shape=(img.shape[0], img.shape[1]))
    bm[:, :] = img > np.ones_like(img)*threshold

    return bm


def structured_mean(selem:np.array, img:np.array, x:int,y:int):
    # calculate window
    H,W = img.shape[-2:]
    h,w = selem.shape
    i_min = clamp(0,W-1,x-w//2)
    j_min = clamp(0,H,y-h//2)
    i_max = clamp(0,W,x+w//2+1)
    j_max = clamp(0,H,y+h//2+1)
    
    # get image region
    reg = img[j_min:j_max, i_min:i_max]
    selem = selem[:reg.shape[0],:reg.shape[1]]
    
    # mask region with selem
    reg *= selem
    
    return reg.sum()/selem.sum()

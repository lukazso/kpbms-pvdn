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


def create_saliency(delta=10):
    # load image and kps
    img = cv2.imread("scripts/data/saliency/072447.png", cv2.IMREAD_GRAYSCALE)
    with open("scripts/data/saliency/072447.json", "r") as f:
        insts = json.load(f)["annotations"][0]["instances"]

    # setup data
    img = np.array(img)
    h, w = img.shape[:2]

    s_maps = []
    
    for inst in insts:
        theta = np.arange(0, 255, delta)
        
        ### 1. step create boolean maps
        b_maps = np.zeros(shape=(theta.shape[0],h,w), dtype=bool)
        
        # loop over all instances in image
        ix,iy = inst["pos"]
        direct = inst["direct"]
                
        # sample intensity value around position of instance
        val = structured_mean(disk(5), img, ix, iy)
        
        # create boolean masks
        for i,t in enumerate(theta):
            if direct:
                b_map =  compute_bm(0.95*val,1.2*val,t,img)
            else:
                b_map =  compute_bm(0.9*val,1*val,t,img)

            # flood fill algorithm from each keypoint
            mask = (flood(b_map,(iy,ix)))
            b_map = np.logical_and(b_map,mask)
            
            # update overall b map
            b_maps[i] = np.logical_or(b_map,b_maps[i])
                    
        # get mean saliency maps
        s_map = np.mean(b_maps, axis=0)
        s_maps.append(s_map)
    
    return s_maps, insts, img



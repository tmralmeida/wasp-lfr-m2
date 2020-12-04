import numpy as np
from .dataset import BoundingBox

def get_roi(bbox, delta = 1.5, min_val = 18):
    """
    Computes the region of interest to find the target
    """
    d_x = max(bbox.width*(delta-1)/2, min_val)
    d_y = max(bbox.height*(delta-1)/2, min_val)
    w = int(bbox.width + 2 * d_x)
    h = int(bbox.height + 2 * d_y)
    search_window_tl = (bbox.xpos - d_x, bbox.ypos - d_y)
    search_window_br = (bbox.xpos + w - d_x, bbox.ypos + h - d_y)
    
    return  BoundingBox("tl-size", int(search_window_tl[0]), int(search_window_tl[1]), w, h)

def normalize(img_cropped):
    """
    Patch normalization
    """
    img_cropped /= 255
    return (img_cropped - np.mean(img_cropped)) / np.std(img_cropped)


def smooth_edge(img):
    """ 
    Smoothing edges, following the paper
    """
    h_width = np.hanning(img.shape[0]) # img.shape --> [h,w]
    h_height = np.hanning(img.shape[1])
    mask = np.sqrt(np.outer(h_width, h_height))
    return img * mask

def transf2ori(loc, bbox, roi, img_s):
    """
    Transform the output of the tracker into the original coordinates
    """
    rows, cols = loc # in relation to the roi
    dx = cols - int(roi.width / 2)
    dy = rows - int(roi.height / 2)
    bbox.xpos += dx
    bbox.ypos += dy
    r_max = img_s[0] - 1
    c_max = img_s[1] - 1
    
    bbox.xpos = np.clip(bbox.xpos, 0, c_max)
    bbox.ypos = np.clip(bbox.ypos, 0, r_max)
    
    roi.xpos += dx
    roi.ypos += dy
    roi.xpos = np.clip(roi.xpos,  0, c_max)
    roi.ypos = np.clip(roi.ypos,  0, r_max)
    return bbox, roi
    
    
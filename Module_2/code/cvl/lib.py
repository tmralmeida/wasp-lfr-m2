import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser 
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
    h_h = np.hanning(img.shape[0]) # img.shape --> [h,w]
    h_w = np.hanning(img.shape[1])
    mask = np.sqrt(np.outer(h_h, h_w))
    return img * mask

def get_2d_gauss(shape, sig = 2, min = -10, max = 10):
    """ 
    Centered 2D gaussian based on https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/
    """
    x, y = np.meshgrid(np.linspace(-10, 10, shape[1]),
                        np.linspace(-10, 10, shape[0]))
    dst = np.sqrt (x**2 + y**2)
    c = np.exp(-((dst**2) / (2.0 * sig**2 )))
    return c
    

def transf2ori(loc, bbox, roi, img_s):
    """
    Transform the output of the tracker into the original coordinates
    """
    rows, cols = loc # in relation to the roi
    dx = cols - int(roi.width / 2)
    dy = rows - int(roi.height / 2)
    r_max = img_s[0] - 1
    c_max = img_s[1] - 1
    
    bbox.xpos += dx
    bbox.ypos += dy
    bbox.xpos = np.clip(bbox.xpos, 0, c_max)
    bbox.ypos = np.clip(bbox.ypos, 0, r_max)
    roi.xpos += dx
    roi.ypos += dy
    roi.xpos = np.clip(roi.xpos,  0, c_max)
    roi.ypos = np.clip(roi.ypos,  0, r_max)
    return bbox, roi
    

def resume_performance(ds, SEQUENCE_IDX, bboxes):
    """ Plots IOU and AUC
    """
    ious = ds.calculate_per_frame_iou(SEQUENCE_IDX, bboxes)
    auc = ds.calculate_auc(SEQUENCE_IDX, bboxes)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(ious)
    plt.title("IOU variation")
    plt.subplot(1,2,2)
    plt.plot(auc)
    plt.title("AUC")
    plt.show()
    

def get_arguments():
    """ Determines each command lines input and parses them
    """
    parser = ArgumentParser()
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default="mosse",
        choices=["mosse", "deep_f", "hand_f"],
        help = "Mosse that you want to run. Options: grayscale mosse (mosse), Deep Features w/ multiscale mosse (deep_f), Handcrafted features w/ multichannels mosse (hand_f)"
    )
        
    parser.add_argument(
        "--ds_idx",
        "-ds",
        type = int,
        default=0,
        help="Dataset indice. See dataset.py file"
    )
    
    parser.add_argument(
        "--show_results",
        "-res",
        type=str2bool,
        default=True,
        help="Show numerical results"
    )
    
    parser.add_argument(
        "--show_viz",
        "-viz",
        type=str2bool,
        default=True,
        help="Show visual results"
    )
    
    parser.add_argument(
        "--ds_path",
        "-path",
        default="Mini-OTB",
        help="Dataset Path"
    )
    
    parser.add_argument(
        "--wait_time",
        "-wt",
        type=int,
        default=25,
        help="0 to press a key in each frame or 25 to wait 25ms"
    )
    
    
    # Hyperparameters for Grayscale MOSSE:
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.125,
        help="Learning rate value; Default: 0.125 (paper)"
    )
    
    parser.add_argument(
        "--lambda",
        type=int,
        default=1e-5,
        help="Regularization parameter;"
    )
    
    return parser.parse_args()

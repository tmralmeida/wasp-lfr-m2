import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser 
from .dataset import BoundingBox


def get_roi(bbox, delta = 1.5, min_val = 18, squared = False, bigger = False):
    if bigger:
        d_x = max(bbox.width*(delta-1)/2, min_val)
        d_y = max(bbox.height*(delta-1)/2, min_val)
        w = int(bbox.width + 2 * d_x)
        h = int(bbox.height + 2 * d_y)
        roi_tl = (bbox.xpos - d_x, bbox.ypos - d_y)
        roi =  BoundingBox("tl-size", int(roi_tl[0]), int(roi_tl[1]), w, h)
    else:
        roi = bbox
    if squared:
        max_side = max(roi.width, roi.height)
        if max_side == roi.width:
            dx = (roi.width - roi.height) // 2
            roi_tl = (roi.xpos, roi.ypos - dx)
            w = roi.width
            h = w
        else:
            dy = (roi.height - roi.width) // 2
            roi_tl = (roi.xpos - dy, roi.ypos)
            h = roi.height
            w = h
    return BoundingBox("tl-size", int(roi_tl[0]), int(roi_tl[1]), w, h) 

def normalize(img_cropped):
    """
    Patch normalization
    """
    # https://www.geeksforgeeks.org/log-transformation-of-an-image-using-python-and-opencv/
    c = 255 / np.log(1 + np.max(img_cropped))
    log_image = c * (np.log(img_cropped + 1))
    return (log_image - np.mean(log_image)) / np.std(log_image)


def smooth_edge(img):
    """ 
    Smoothing edges, following the paper
    """
    h_h = np.hanning(img.shape[-2]) 
    h_w = np.hanning(img.shape[-1])
    mask = np.sqrt(np.outer(h_h, h_w))
    return img * mask


def get_2d_gauss(shape, sig = 2, min = -10, max = 10):
    """ 
    Centered 2D gaussian based on https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/
    """
    x, y = np.meshgrid(np.linspace(-10, 10, shape[-1]),
                        np.linspace(-10, 10, shape[-2]))
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
        "--bigger_roi",
        "-br",
        type=str2bool,
        default=True,
        help="Whether use bigger roi or not"
    )
    
    parser.add_argument(
        "--squared_roi",
        "-sc",
        type=str2bool,
        default=True,
        help="Using a squared roi"
    )
    
    parser.add_argument(
        "--tracker_type",
        "-tt",
        type=str,
        default="mosse",
        choices=["mosse", "resnet", "mobilenet", "alexnet", "vgg16", "hog", "cn", "hog_cn"],
        help = "Mosse that you want to run. Options: grayscale mosse (mosse), Deep Features, Handcrafted features (hog, hog_cn, cn)"
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

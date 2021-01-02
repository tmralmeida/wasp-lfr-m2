#!/usr/bin/env python3

import cv2
import numpy as np
from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import MOSSETracker, DCFMOSSETracker
from cvl.lib import get_roi, resume_performance, get_arguments
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import torch

args = get_arguments()
dataset_path = args.ds_path
device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

if __name__ == "__main__":
    dataset = OnlineTrackingBenchmark(dataset_path)
    a_seq = dataset[args.ds_idx]
    
    if args.show_viz: #visual results
        cv2.namedWindow("tracker")
    if args.tracker_type == "mosse":
        tracker = MOSSETracker(lambda_ = args.lambda_,
                               learning_rate = args.learning_rate)
        squared = False
        bigger = args.bigger_roi
    elif args.tracker_type in ["resnet", "mobilenet", "alexnet", "vgg16", "hog"]:
        tracker = DCFMOSSETracker(dev = device, 
                                  features = args.tracker_type,
                                  lambda_ = args.lambda_,
                                  learning_rate = args.learning_rate)
        squared = args.squared_roi
        bigger = args.bigger_roi
    bboxes = []
    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)-1}")
        image_color = frame['image']
        if args.tracker_type == "mosse":
            image = np.sum(image_color, 2) / 3 # grayscale
        else:
            image = np.transpose(np.float64(image_color), (2, 0, 1))
        if frame_idx == 0:
            bbox = copy(frame['bounding_box'])
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1
            roi = get_roi(bbox, squared = squared, bigger = bigger) # get roi slightly bigger than bbox
            tracker.start(image, bbox, roi) # first frame approach
        else:
            tracker.detect(image)
            tracker.update()
        
        bboxes.append(copy(tracker.bbox))
        if args.show_viz:
            bbox = tracker.bbox
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 0, 255), thickness=3)
            cv2.imshow("tracker", image_color)
            if cv2.waitKey(args.wait_time) == ord("s"):
                cv2.imwrite(f"../results/imgs/ds_{args.ds_idx}-img_{frame_idx}.png", image_color)
                print("image saved")
            
    cv2.destroyAllWindows()
    if args.show_results: #plotting iou, auc
        print("Resuming results....")
        resume_performance(dataset, args.ds_idx, bboxes)
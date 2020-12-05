#!/usr/bin/env python3

import cv2
import numpy as np
from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import MOSSETracker
from cvl.lib import get_roi, resume_performance, get_arguments
import matplotlib.pyplot as plt
from copy import copy, deepcopy


args = get_arguments()

dataset_path = args.ds_path

if __name__ == "__main__":
    dataset = OnlineTrackingBenchmark(dataset_path)
    a_seq = dataset[args.ds_idx]
    
    if args.show_viz: #visual results
        cv2.namedWindow("tracker")
    tracker = MOSSETracker()
    bboxes = []
    
    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)-1}")
        image_color = frame['image']
        image = np.sum(image_color, 2) / 3 # grayscale
        if frame_idx == 0:
            bbox = copy(frame['bounding_box'])
            aaaa = bbox
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1
            roi = get_roi(bbox) # get roi slightly bigger that bbox
            tracker.start(image, bbox, roi) # first frame approach
        else:
            tracker.detect(image)
            tracker.update(image)
        
        # print(tracker.bbox)
        if args.show_viz:
            bboxes.append(copy(tracker.bbox))
            bbox = tracker.bbox
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow("tracker", image_color)
            cv2.waitKey(0)
            
    cv2.destroyAllWindows()
    if args.show_results: #plotting iou, auc
        print("Resuming results....")
        resume_performance(dataset, args.ds_idx, bboxes)
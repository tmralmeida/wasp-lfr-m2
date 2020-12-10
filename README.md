# wasp-lfr-m2

Code for Learning Features Representation WASP course module 2.

## Input arguments 

| Argument Name       |       Type       |      Default       |       Description      |
|---------------------|:----------------:|:------------------:|------------------------|
| --tracker_type      |     `string`     |   `"mosse"`        | Type of Tracker(lib.py)|
| --ds_idx            |     `int`        |   `0`              | Dataset idx(dataset.py)|  
| --show_results      |     `bool`       |  `True`            | Plot results           |
| --show_viz          |     `bool`       | `True`             | See the results        |
| --ds_path           |     `string`     |                    | Dataset dir path       |
| --wait_time         |     `int`        | `25`               | Time between each frame|
| --learning_rate     |     `int`        | `0.125`            | Learning rate value    |
| --lambda            |     `int`        | `1e-5`             | Lambda (reg.) value    |

Run e.g.:


````
python run_tracker.py --tracker_type resnet --ds_idx 4 --wait time 0
````

This would run dataset 4 with deep features (from a Resnet layer) in the multi-channel MOSSE, and the
user have to press any key to continue to the next frame (--wait_time 0).
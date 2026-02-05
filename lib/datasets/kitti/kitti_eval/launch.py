import time
import fire
import os
import json

from . import utils as kitti
from .core import get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    
    results_str, results_dict, rich_data = get_official_eval_result(gt_annos, dt_annos, current_class)
    
    # Try to load previous results for comparison
    prev_rich_data = None
    try:
        # result_path usually ends with something like .../epoch_xx/data or .../epoch_xx
        # We want to store the "last_result.json" in the parent directory of the current result folder
        # to share it across different epochs if they are in subfolders.
        # Assuming structure: experiments/run_id/visualizations/epoch_X/
        # We want to store in: experiments/run_id/visualizations/last_eval_result.json
        
        # If result_path is '.../epoch_1', parent is '.../visualizations'
        # If result_path is '.../epoch_1/data', parent is '.../epoch_1', parent.parent is '.../visualizations'
        
        # Heuristic: go up until we find a 'visualizations' folder or just go up 2 levels
        # Safe bet: Just go up 2 levels from the folder containing the txt files.
        parent_dir = os.path.dirname(os.path.normpath(result_path))
        if os.path.basename(parent_dir) == 'data': # Handle if inside 'data' folder
             parent_dir = os.path.dirname(parent_dir)
             
        # Go up one more level to exit "epoch_X" folder usually
        base_dir = os.path.dirname(parent_dir) 
        
        last_result_path = os.path.join(base_dir, 'last_eval_result.json')
        
        if os.path.exists(last_result_path):
            with open(last_result_path, 'r') as f:
                prev_rich_data = json.load(f)
        
        # Save current results as last results for next time
        import numpy as np
        with open(last_result_path, 'w') as f:
            json.dump(rich_data, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else None)
    except Exception as e:
        # print(f"Warning: Could not handle previous result file: {e}")
        pass

    from lib.helpers.logger_helper import print_kitti_eval_results
    print_kitti_eval_results(rich_data, prev_rich_data)
    return results_str


if __name__ == '__main__':
    fire.Fire(evaluate)

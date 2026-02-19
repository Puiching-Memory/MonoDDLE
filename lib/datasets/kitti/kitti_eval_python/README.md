# kitti-object-eval-python
**Note**: This is borrowed from [traveller59/kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python) with minor modification (add an interface to evaluate the results based on their distance range).

Fast kitti object detection eval in python(finish eval in less than 10 second), support 2d/bev/3d/aos. , support coco-style AP.
## Dependencies
Only support python 3.6+, need `numpy`, `skimage`, `fire`.
## Usage
* commandline interface:
```
python evaluate.py evaluate --label_path=/path/to/your_gt_label_folder --result_path=/path/to/your_result_folder --label_split_file=/path/to/val.txt --current_class=0 --coco=False
```
* python interface:
```Python
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/path/to/your_result_folder"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/path/to/your_gt_label_folder"
gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer
```

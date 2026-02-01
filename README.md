# Delving into Localization Errors for Monocular 3D Detection

## Introduction

## Usage

### Installation
This repo is tested on our local environment (python=3.13, cuda=12.8, pytorch=2.9.1), and we recommend you to use uv to create a virtual environment:

```bash
uv venv --python=3.13
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```text
#ROOT
└── data
    └── KITTI
        ├── ImageSets [already provided]
        └── object
            ├── training
            │   ├── calib (unzipped from calib.zip)
            │   ├── image_2 (unzipped from left_color.zip)
            │   └── label_2 (unzipped from label_2.zip)
            └── testing
                ├── calib
                └── image_2
```

### Training & Evaluation

Run the following commands in the project root:

```sh
# 1. Training (using default config)
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml

# 2. Evaluation only
python tools/train_val.py --config experiments/kitti/monodle_kitti.yaml --e

# 3. Check all available options
python tools/train_val.py -- --help
```

The model will be evaluated automatically once training is completed. You can run `bash experiments/kitti/clear_cache.sh` to quickly remove logs and checkpoints.

For ease of use, we also provide a pre-trained checkpoint, which can be used for evaluation directly. See the below table to check the performance.

|                   | AP40@Easy | AP40@Mod. | AP40@Hard |
| ----------------- | --------- | --------- | --------- |
| In original paper | 17.45     | 13.66     | 11.68     |
| In this repo      | 17.94     | 13.72     | 12.10     |

## Citation

If you find our work useful in your research, please consider citing:

```latex
```

## Acknowlegment

This repo benefits from the excellent work [CenterNet](https://github.com/xingyizhou/CenterNet). Please also consider citing it.

## License

This project is released under the GNU General Public License v3.0 (GPL-3.0).

## Contact

If you have any questions about this project, please feel free to contact 1138663075@qq.com.

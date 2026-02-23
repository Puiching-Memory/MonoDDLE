import os
import argparse
import glob
import sys
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Add third_party to path if depth_anything_3 is not installed
script_dir = os.path.dirname(os.path.abspath(__file__))
monoddle_root = os.path.dirname(script_dir)
da3_path = os.path.join(monoddle_root, 'third_party', 'Depth-Anything-3', 'src')
if os.path.exists(da3_path) and da3_path not in sys.path:
    sys.path.append(da3_path)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Could not import depth_anything_3. Please clarify installation or check path.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate DA3 depth maps for KITTI dataset')
    parser.add_argument('--data_path', type=str, default='data/KITTI', help='Path to KITTI dataset root')
    parser.add_argument('--save_path', type=str, default='data/KITTI/DA3_depth_results', help='Path to save depth results')
    parser.add_argument('--model_name', type=str, default='depth-anything/DA3Metric-Large', help='Pretrained model name')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'testing'], help='Split to process')
    return parser.parse_args()

def main():
    args = parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        model = DepthAnything3.from_pretrained(args.model_name).to(device)
    except Exception as e:
        print(f"Error loading model {args.model_name}: {e}")
        print("Falling back to generic load or check check instructions.")
        return

    # Prepare directories
    image_dir = os.path.join(args.data_path, args.split, 'image_2')
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    # Get images
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    print(f"Found {len(image_files)} images in {image_dir}")

    # Inference loop
    # We process one by one to avoid OOM on large images and to save individually
    for img_path in tqdm(image_files):
        try:
            # Predict
            # DA3 inference returns a Prediction object with .depth, .intrinsics, .extrinsics
            prediction = model.inference([img_path])
            depth = prediction.depth[0]  # [H, W]

            # Extract intrinsics (3, 3) and extrinsics (4, 4) -> (3, 4)
            save_data = dict(depth=depth)
            if prediction.intrinsics is not None:
                save_data['intrinsics'] = prediction.intrinsics[0].astype(np.float32)
            if prediction.extrinsics is not None:
                # Store first 3 rows of 4x4 matrix as (3, 4)
                save_data['extrinsics'] = prediction.extrinsics[0][:3, :].astype(np.float32)

            # Save
            file_name = os.path.basename(img_path)
            file_id = os.path.splitext(file_name)[0]
            save_file = os.path.join(save_dir, f"{file_id}.npz")
            np.savez_compressed(save_file, **save_data)

            # Generate visualization: original image resized + colorized depth, stacked vertically
            h, w = depth.shape
            orig_img = cv2.imread(img_path)
            orig_resized = cv2.resize(orig_img, (w, h))

            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

            vis = np.vstack([orig_resized, depth_color])
            vis_file = os.path.join(save_dir, f"{file_id}_vis.jpg")
            cv2.imwrite(vis_file, vis)

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    print("Done.")

if __name__ == '__main__':
    main()

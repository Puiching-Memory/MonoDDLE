import os
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

def process_kitti_depth():
    # 1. 配置路径 (基于项目根目录自动计算)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "third_party/Depth-Anything-3/models")
    input_dir = os.path.join(base_dir, "data/KITTI/training/image_2")
    output_dir = os.path.join(base_dir, "data/KITTI/DA3_depth_results")
    
    os.makedirs(output_dir, exist_ok=True)

    # 2. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型: {model_path} ...")
    model = DepthAnything3.from_pretrained(model_path).to(device)
    model.eval()

    # 3. 获取所有图像
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"找到 {len(image_paths)} 张图像，开始处理...")

    # 4. 逐帧推理并保存
    # 注意：为了节省显存，我们单张处理图像
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # 目标文件路径 (不再为每个图像创建文件夹)
        npz_path = os.path.join(output_dir, f"{base_name}.npz")
        vis_path = os.path.join(output_dir, f"{base_name}_vis.jpg")
        
        # 断点续传：如果 npz 已存在则跳过
        if os.path.exists(npz_path):
            continue

        try:
            # 执行推理 (不使用自动导出，以便自定义文件名)
            prediction = model.inference(
                image=[img_path],
                process_res=504,
                export_dir=None # 禁止自动分文件夹导出
            )
            
            # 手动保存原始深度数据 (mini_npz 格式)
            # 这里的字段名与官方 mini_npz 保持一致: depth, intrinsics, extrinsics
            np.savez_compressed(
                npz_path,
                depth=prediction.depth[0],
                intrinsics=prediction.intrinsics[0],
                extrinsics=prediction.extrinsics[0]
            )
            
            # 手动生成并保存可视化图
            depth_vis = visualize_depth(prediction.depth[0])
            # 获取原始图像用于拼接 (prediction.processed_images 已经是 uint8 [H, W, 3])
            raw_img = prediction.processed_images[0]
            
            # 上下拼接 (vstack)
            combined_vis = np.vstack([raw_img, depth_vis])
            
            # 这里 combined_vis 是 RGB，cv2 需要 BGR
            cv2.imwrite(vis_path, cv2.cvtColor(combined_vis, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"处理 {img_name} 时出错: {e}")
            continue

    print(f"处理完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    process_kitti_depth()

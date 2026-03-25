import os
import sys
import cv2
import numpy as np
import torch

# 确保能导入 lib 下的模块
# 当前脚本在 lib/data/ 下，所以往上跳两级到项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from lib.data.water_dataset import WaterDataset

# ================= 配置区域 =================

# 数据集根目录 (包含 images 和 labels 文件夹的上级目录，或者直接是 images 的父级)
# 根据您的报错历史，您的结构应该是:
# D:\...\datasets\water_seg\
#       ├── images\  (或 img)
#       ├── labels\  (或 mask)
#       └── train.txt
DATASET_ROOT = r'D:\Files\GitProject\BiSeNet-LY\datasets\water_seg'

# 标注文件路径 (对应参数 annpath)
# 注意：water_dataset.py 内部可能会用这个文件里的相对路径去拼接 im_root
ANN_PATH = os.path.join(DATASET_ROOT, 'train.txt')

print("🔍 开始验证数据加载...")
print(f"图片根目录 (im_root): {DATASET_ROOT}")
print(f"标注文件路径 (annpath): {ANN_PATH}")

# ================= 验证逻辑 =================

try:
    # 【关键修正】使用正确的参数名: im_root, annpath
    dataset = WaterDataset(
        im_root=DATASET_ROOT, 
        annpath=ANN_PATH, 
        trans_func=None,
        mode='train'
    )
    
    print(f"✅ 数据集初始化成功！样本数量: {len(dataset)}")
    print("-" * 40)

    if len(dataset) == 0:
        print("❌ 警告：数据集为空！请检查 train.txt 内容或路径。")
    else:
        prev_pixels = -1
        
        # 检查前 5 张图片
        for i in range(min(5, len(dataset))):
            try:
                im, lb = dataset[i]
                
                # 统计水域像素 (值为 1 的像素)
                water_pixels = torch.sum(lb).item()
                total_pixels = lb.numel()
                ratio = water_pixels / total_pixels * 100
                
                print(f"Sample {i}: 水域像素={water_pixels:,}, 占比={ratio:.2f}%")
                
                # 简单校验
                if water_pixels == 0:
                    print("  ❌ 错误：这张图的标签全是背景(0)，没读到红色水域！")
                    print("     -> 这通常意味着 water_dataset.py 中的颜色阈值 (cv2.inRange) 需要调整。")
                elif i > 0:
                    diff = abs(water_pixels - prev_pixels)
                    if diff < 500: 
                        print(f"  ⚠️ 警告：与上一张图差异过小 ({diff})，可能读取逻辑有误！")
                    else:
                        print("  ✅ 正常：水域面积有显著变化。")
                
                prev_pixels = water_pixels
                
            except Exception as e:
                print(f"Sample {i} 处理出错: {e}")
                import traceback
                traceback.print_exc()

    print("-" * 40)
    print("✅ 验证完成。")
    print("👉 如果看到 '水域像素' 数值各不相同且不为 0，说明数据读取修复成功！")
    print("👉 接下来请重新运行训练脚本。")

except FileNotFoundError as fe:
    print(f"❌ 文件未找到: {fe}")
    print("提示：请检查 DATASET_ROOT 或 train.txt 路径是否存在。")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")
    import traceback
    traceback.print_exc()
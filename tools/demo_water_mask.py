import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
# 兼容不同版本的 torch.load
try:
    state_dict = torch.load(args.weight_path, map_location='cpu')
except Exception:
    state_dict = torch.load(args.weight_path, map_location='cpu', weights_only=False)
    
net.load_state_dict(state_dict, strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)

# 1. 读取原始图像 (BGR格式，用于后续合成)
im_orig = cv2.imread(args.img_path)
if im_orig is None:
    raise FileNotFoundError(f"Image not found: {args.img_path}")

# 转换为RGB用于模型输入
im_rgb = im_orig[:, :, ::-1] 

im = to_tensor(dict(im=im_rgb, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out = net(im)[0]
out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
out = out.argmax(dim=1)

# --- 可视化部分修改开始 ---

# 获取预测结果 (H, W)，值为 0 或 1
pred_mask = out.squeeze().detach().cpu().numpy().astype(np.uint8)

# 1. 创建红色蒙版 (BGR格式: 0, 0, 255)
# 形状必须和原图一致 (H, W, 3)
red_overlay = np.zeros_like(im_orig)
red_overlay[:, :, 0] = 0   # B
red_overlay[:, :, 1] = 0   # G
red_overlay[:, :, 2] = 255 # R

# 2. 创建透明度通道 (alpha)
# 水域区域 (pred_mask == 1) 透明度设为 0.5 (50%透明)
# 背景区域 (pred_mask == 0) 透明度设为 0 (完全显示原图)
alpha = np.zeros_like(pred_mask, dtype=np.float32)
alpha[pred_mask == 1] = 0.5  # 您可以调整这个值 (0.1 ~ 0.8) 来改变红色的深浅

# 扩展alpha维度以匹配图像通道 (H, W) -> (H, W, 1)
alpha_3ch = alpha[:, :, np.newaxis]

# 3. 融合图像
# 公式: Result = Original * (1 - alpha) + Overlay * alpha
# 注意转换为 float 进行计算，避免溢出
im_orig_float = im_orig.astype(np.float32)
red_overlay_float = red_overlay.astype(np.float32)

final_image = im_orig_float * (1 - alpha_3ch) + red_overlay_float * alpha_3ch
final_image = final_image.astype(np.uint8)

# 保存结果
cv2.imwrite('./res_visual.jpg', final_image)
print("Visualization saved to ./res_visual.jpg")

# 同时也保存原始的纯色mask图 (可选，方便调试)
pred_color = palette[pred_mask]
cv2.imwrite('./res_mask_only.jpg', pred_color)
print("Mask only saved to ./res_mask_only.jpg")

# --- 可视化部分修改结束 ---
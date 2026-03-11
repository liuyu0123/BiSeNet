import sys
sys.path.insert(0, '.')
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py')
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth')
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png')
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)

# define model
print(f"Loading model: {cfg.model_type}")
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')

# --- 权重加载 ---
print(f"Loading weights from: {args.weight_path}")
try:
    state_dict = torch.load(args.weight_path, map_location='cpu')
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    
    if len(pretrained_dict) == 0:
        raise RuntimeError("❌ 错误：加载的权重文件中没有任何键与当前模型匹配！")
    
    net.load_state_dict(pretrained_dict, strict=True)
    print("✅ 权重加载成功！")
except Exception as e:
    print(f"❌ 权重加载失败: {e}")
    sys.exit(1)

net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223),
    std=(0.2112, 0.2148, 0.2115),
)

# 读取原始图像
im_orig = cv2.imread(args.img_path)
if im_orig is None:
    raise FileNotFoundError(f"Image not found: {args.img_path}")

im_rgb = im_orig[:, :, ::-1] 
im = to_tensor(dict(im=im_rgb, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

# inference
print(f"Input shape: {im.shape}")
im_resized = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out_logits = net(im_resized)[0]
out_logits = F.interpolate(out_logits, size=org_size, align_corners=False, mode='bilinear')
out = out_logits.argmax(dim=1)

# 获取预测结果
pred_mask = out.squeeze().detach().cpu().numpy().astype(np.uint8)
unique_vals, counts = np.unique(pred_mask, return_counts=True)
print(f"📊 当前图片预测结果统计: {dict(zip(unique_vals, counts))}")

# --- 可视化部分 ---

# 1. 形态学平滑 (可选，去除噪点)
kernel = np.ones((3,3),np.uint8)
pred_mask_smoothed = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

# 2. 创建红色蒙版 (BGR: 0, 0, 255)
water_color = np.array([0, 0, 255], dtype=np.uint8) # 红色
colored_overlay = np.zeros_like(im_orig)
colored_overlay[pred_mask_smoothed == 1] = water_color

# 3. 融合
alpha_val = 0.4
alpha_channel = (pred_mask_smoothed == 1).astype(np.float32) * alpha_val
alpha_3ch = alpha_channel[:, :, np.newaxis]

im_orig_float = im_orig.astype(np.float32)
colored_overlay_float = colored_overlay.astype(np.float32)

final_image = im_orig_float * (1 - alpha_3ch) + colored_overlay_float * alpha_3ch
final_image = final_image.astype(np.uint8)

# 生成文件名前缀
file_name = args.img_path.split('\\')[-1].split('.')[0]

# === 保存 1: 叠加效果图 (红色) ===
output_visual = f"./result/res_visual_{file_name}.jpg"
cv2.imwrite(output_visual, final_image)
print(f"✅ 叠加效果已保存至: {output_visual}")

# === 保存 2: 纯 Mask 图 (黑白) ===
# 将 0/1 映射为 0/255 以便查看
mask_255 = (pred_mask_smoothed * 255).astype(np.uint8)
output_mask = f"./result/res_mask_only_{file_name}.jpg"
cv2.imwrite(output_mask, mask_255)
print(f"✅ 纯 Mask 已保存至: {output_mask}")
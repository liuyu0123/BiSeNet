import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import os

class WaterDataset(Dataset):
    def __init__(self, im_root, annpath, trans_func=None, mode='train'):
        self.im_root = im_root
        self.mode = mode
        self.trans_func = trans_func
        
        with open(annpath, 'r') as f:
            # 每一行格式：图片相对路径，标签相对路径
            self.anns = [line.strip().split(',') for line in f.readlines()]
        
        self.n_cats = 2
        self.lb_ignore = 255
        
        print(f'Water {mode} set: {len(self.anns)} samples')
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        impath, lbpath = self.anns[idx]
        
        # 1. 读取图像 (RGB)
        im = cv2.imread(os.path.join(self.im_root, impath))
        if im is None:
            raise FileNotFoundError(f"Cannot read image: {impath}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # 2. 【修正】读取标签并提取深红色区域
        lb_color = cv2.imread(os.path.join(self.im_root, lbpath), cv2.IMREAD_COLOR)
        if lb_color is None:
            raise FileNotFoundError(f"Cannot read label: {lbpath}")
        
        # --- 核心修复：调整深红色阈值 ---
        # 注意：cv2.imread 读出来是 BGR 格式
        # 您的标签看起来是深红色，B 和 G 很低，R 较高 (例如 R=139, G=0, B=0)
        # 我们设置一个宽松的范围来捕获这种深红色
        lower_red = np.array([0, 0, 50], dtype=np.uint8)   # B, G, R (允许 R 从 50 开始)
        upper_red = np.array([60, 60, 200], dtype=np.uint8) # B, G, R (允许 R 到 200，防止过曝)
        
        mask_temp = cv2.inRange(lb_color, lower_red, upper_red)
        
        # 将掩码转为 0/1 标签
        lb = (mask_temp > 0).astype(np.uint8)
        
        # 【调试用】打印前几张图的统计信息，确认读取正确
        if idx < 3:
            print(f"Debug [{idx}]: Label path={lbpath}, Water pixels={np.sum(lb)}, Shape={lb.shape}")
            
        # 如果读出来全是 0，说明阈值还是不对，给出警告
        if np.sum(lb) == 0:
            print(f"⚠️ 警告：图片 {lbpath} 未检测到红色水域！请检查标签颜色或阈值。")
            # 可以选择跳过或强制处理，这里先保留全 0 让训练报错或观察
        
        # 3. 数据增强
        if self.trans_func is not None:
            result = self.trans_func({'im': im, 'lb': lb})
            im = result['im']
            lb = result['lb']
        
        # 4. 转为 Tensor
        im = torch.from_numpy(im.copy()).permute(2, 0, 1).float()
        lb = torch.from_numpy(lb.copy()).long()
        
        return im, lb
"""
    本代码用于: 定义训练过程和测试过程
    创建时间: 2021 年 11 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 26 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- #
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 导入自定义的包 ----- #
from dataset import NERdataset


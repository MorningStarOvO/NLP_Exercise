"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2021 年 12 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
from pprint import pprint

# ----- 模型创建相关的 ----- #
import torch
import torch.optim as optim 
import torch.nn as nn

# ----- 数据处理相关的 ----- #
import numpy as np
from numpy import save 
import scipy.spatial as T

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 命令行交互相关的 ----- #
import argparse
import logging

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle

# ----- 文件读取相关的 ----- #
import json 

# ----- 画图相关的 ----- #
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='SimHei.ttf', size=16)    # 创建字体对象



# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
# ---------- 设置随机种子, 使结果可复现 ---------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 

def set_logger(log_path):
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s : %(message)s'))
        logger.addHandler(stream_handler)




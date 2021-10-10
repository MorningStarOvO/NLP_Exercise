"""
    本代码用于: XXXX
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
import pprint

# ----- 模型创建相关的 ----- #
import torch

# ----- 数据处理相关的 ----- #
import numpy as np

# ----- 可视化运行过程 ----- #
import tqdm

# ----- 命令行交互相关的 ----- #
import argparse

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle


# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关 ----- # 


# ----- 存储文件相关 ----- #



# ==================== 函数实现 ==================== #
# ----- 设置随机种子, 使结果可复现 ----- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 




# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
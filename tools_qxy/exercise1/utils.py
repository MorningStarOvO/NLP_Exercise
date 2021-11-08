"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
    具体的函数: 
        setup_seed(): 置随机种子, 使结果可复现 
        adjust_learning_rate(): 动态调整学习率
        draw_acc_pic(): 汇总训练、测试、损失曲线
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
import pprint

# ----- 模型创建相关的 ----- #
import torch
import torch.nn.functional as F  
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.optim as optim

# ----- 自定义的模型创建相关的包 ----- #
from dataset import Dataloader
from model import DNN, CNN, RNN

# ----- 图像读取相关 ----- #
# import cv2 
from PIL import Image

# ----- 数据处理相关的 ----- #
import numpy as np

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 命令行交互相关的 ----- #
import argparse

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle

# ----- 画图相关 ----- #
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='SimHei.ttf', size=16)    # 创建字体对象

# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关 ----- # 
PATH_TRAIN_MINI = "data/exp1_train"
PATH_TEST_MINI = "data/exp1_val"

# ----- 存储文件相关 ----- #
PATH_CHECKPOINT = "checkpoint/exp1" 
PATH_LOGS = "logs/exp1"

PATH_PIC = "output_pic/exp1"

# ==================== 函数实现 ==================== #
# ---------- 设置随机种子, 使结果可复现 ---------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 

# ----- 动态调整学习率 ----- #
def adjust_learning_rate(optimizer, epoch, args):
    """
    设置学习率
    :param optimizer: 优化器
    :param epoch: 轮数
    :param args: 命令行的输入
    :return:
    """

    # 初始学习率等于从命令行中输入的
    lr = args.lr

    # 随着训练轮数的增加，学习率不断下降
    if 20 < epoch <= 30: # 20 到 30 轮次学习率降低
        lr = 1e-5
    elif 30 < epoch : # 超过 30 轮次学习率更低
        lr = 1e-6

    # 赋值学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 实时打印当前的学习率
    print("learning rate -> {}".format(lr))


# ----- 绘制曲线 ----- # 
def draw_acc_pic(acc_train_list, acc_test_list, loss_list, args):
    """
    绘制曲线
        acc_train_list: 训练集的 acc 的列表
        acc_test_list: 测试集的 acc 列表
        loss_list: 损失值的列表
    """
    # 绘制损失值曲线
    plt.figure()
    plt.plot(loss_list, label="loss")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("损失值", fontproperties=font)
    str_title = args.model + " 损失值曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()
    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))

    # 汇总训练集和测试集的准确率
    plt.figure()
    plt.plot(acc_train_list, 'ro-', label="train acc")
    plt.plot(acc_test_list, 'bo-', label="test acc")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("准确率", fontproperties=font)
    str_title = args.model + " 准确率曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()
    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))

    # 保存 acc 值
    acc_test_list = np.array(acc_test_list)
    np.save(os.path.join(PATH_PIC, args.model + " 测试集准确率.npy"), acc_test_list)
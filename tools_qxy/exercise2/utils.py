"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 14 日
    具体的函数:
        setup_seed(): 置随机种子, 使结果可复现 
        adjust_learning_rate(): 动态调整学习率
        
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
from pprint import pprint

# ----- 模型创建相关的 ----- #
import torch

# ----- 数据处理相关的 ----- #
import numpy as np

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 命令行交互相关的 ----- #
import argparse

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle

# ----- 文件读取相关的 ----- #
import json 

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
# ---------- 设置随机种子, 使结果可复现 ---------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 

# ---------- 动态调整学习率 ---------- #
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

# ---------- 文本数据的预处理 ---------- #
def txt_preprocess(path_read, path_save):
    print("=> 文本预处理！")

    # ----- 读取 TXT ----- #
    with open(path_read, 'r') as f:
        lines = f.readlines()
        # print(type(lines)) # list

        # 求取列表中最长的字符串
        # max_str = max(lines, key=len, default='')
        # max_len = len(max_str)
        # print("最长的字符串为: ", max_str)
        # print("长度为: ", max_len)
        str_len = []
        for line in lines:
            temp_word = line.split()
            str_len.append(len(temp_word))
        str_len = np.array(str_len)
        max_len = str_len.max()
        print("最长的字符串长度为: \n", max_len)

        # 建立词汇表
        word_list = " ".join(lines).split() # 以 " " 连接各个句子, 并以空格分离
        word_list = list(set(word_list))
        
        # 建立字典 (one-hot 编码)
        word_dict = {w:i for i, w in enumerate(word_list)}
        number_dict = {i:w for i, w in enumerate(word_list)}
        n_class = len(word_dict)

        # 保存字典
        temp = json.dumps(word_dict, indent=4, ensure_ascii=False)
        with open(path_save + "word_dict.json", 'w') as f_write:
            f_write.write(temp)

        temp2 = json.dumps(number_dict, indent=4, ensure_ascii=False)
        with open(path_save + "number_dict.json", 'w') as f_write:
            f_write.write(temp2)
        
        return word_dict, number_dict, n_class, max_len


# ---------- 建立词组对字典 ---------- #



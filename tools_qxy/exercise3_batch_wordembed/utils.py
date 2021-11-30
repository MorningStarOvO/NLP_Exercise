"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2021 年 11 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 26 日
    具体的函数: 
        setup_seed(): 置随机种子, 使结果可复现 
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
import pprint

# ----- 图像读取相关 ----- #
# import cv2 
from PIL import Image

# ----- 模型创建相关的 ----- #
import torch

# ----- 数据处理相关的 ----- #
import numpy as np
import pandas as pd 

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


# ----- 存储文件相关 ----- #

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# ==================== 函数实现 ==================== #
# ---------- 设置随机种子, 使结果可复现 ---------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 

# ---------- config for training ---------- # 
class Config():
    def __init__(self):
        self.learning_rate = 1e-4
        self.dropout = 0.9
        self.epoch = 30
        self.data_dir = 'data/exp3/'
        self.embedding_dim = 300
        self.hidden_dim = 512
        self.save_model = 'checkpoint/exp3/model.pth'
        self.batch_size = 128
        self.socre_choice = 0

# ----- 建立 glove 词表 ----- # 
def build_glove():
    word_2_embed = {}
    with open("data/glove/glove.6B.300d.txt", encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]
            temp = np.array(line[1:], dtype=np.float32)
            word_2_embed[word] = temp 

    return word_2_embed

# ---------- 建立词表 ---------- #
def build_vocab(data_dir):
    """
    :param data_dir: the dir of train_corpus.txt
    :return: the word dict for training
    """

    # ----- 如果已经建立词表，则「直接读取」 ----- #
    if(os.path.isfile('word_dict.npy')):
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        return word_dict
    # ----- 建立词表，并保存为 .npy 格式 ----- #
    # 这里直接得到了词向量
    else:
        word_dict = {}
        train_corpus = data_dir + 'train' +'_corpus.txt'
        lines = open(train_corpus).readlines()
        for line in lines:
            word_list = line.split()
            for word in word_list:
                if(word not in word_dict):
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        
        # ----- 对词表进行排序 ----- #
        word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
        
        # ----- 保存为 .npy 格式 ----- #
        np.save('word_dict.npy', word_dict)
        
        # ----- 加载词表 ----- #
        word_dict = np.load('word_dict.npy', allow_pickle=True).item()
        
        return word_dict

# ---------- 建立字典 ---------- #
def build_dict(word_dict):
    """
    :param word_dict:
    :return: word2id and tag2id
    """

    # 7 is the label of pad
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7, START_TAG: 8, STOP_TAG: 9}
    word2id = {}
    for key in word_dict:
        word2id[key] = len(word2id)
    word2id['unk'] = len(word2id)
    word2id['pad'] = len(word2id)
    return word2id, tag2id

# ---------- 计算最大长度 ---------- #
def cal_max_length(data_dir):

    file = data_dir + 'train' + '_corpus.txt'
    lines = open(file).readlines()
    max_len = 0
    for line in lines:
        if(len(line.split()) > max_len):
            max_len = len(line.split())

    return max_len
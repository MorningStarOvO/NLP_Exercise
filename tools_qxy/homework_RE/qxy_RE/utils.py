"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
    具体的函数: 
        setup_seed(): 置随机种子, 使结果可复现 
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
import pprint

# ----- 命令行交互的 ----- #
import logging

# ----- 图像读取相关 ----- #
# import cv2 
from PIL import Image

# ----- 模型创建相关的 ----- #
import torch

# ----- 数据处理相关的 ----- #
import numpy as np
import pandas as pd 
import json 
import shutil

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


# ==================== 函数实现 ==================== #
# ---------- 设置随机种子, 使结果可复现 ---------- #
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True 

class Params():
    """
    Class that loads hyperparameters from a json file.
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by `params.dict['learning_rate']
        """
        return self.__dict__ 


class RunningAverage():
    """
    A simple class that maintains the running average of a quantity
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def generate_zero_vector(embedding_dim):
    return [0] * embedding_dim

def generate_random_vector(embedding_dim):
    return np.random.uniform(-np.sqrt(3.0 / embedding_dim), np.sqrt(3.0 / embedding_dim),
                             embedding_dim).tolist()
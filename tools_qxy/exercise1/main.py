"""
    本代码用于: XXXX
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
    具体步骤: 
        step1: 建立网络模型、数据加载器、优化器、损失函数
        step2: 1 Epoch 的训练和测试
        step3: 画图
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
from dataset import Dataloader  # 导入数据加载器
from model import DNN, CNN, RNN # 导入模型
from utils import setup_seed, adjust_learning_rate, draw_acc_pic # 导入常用函数
from process import train, test # 导入训练、测试过程
from option import parse_opt # 导入命令行交互函数

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


# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关 ----- # 
PATH_TRAIN_MINI = "data/exp1_train"
PATH_TEST_MINI = "data/exp1_val"

# ----- 存储文件相关 ----- #
PATH_CHECKPOINT = "checkpoint/exp1" 
PATH_LOGS = "logs/exp1"

# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step0: 输入 args 命令 ---------- #
    args = parse_opt()

    # 设置随机数种子
    setup_seed(args.random)

    # ---------- step1: 建立网络模型、数据加载器、优化器、损失函数 ---------- #
    # 创建网络模型
    print("=> creating model: ", args.model)
    print('\n')
    if args.model == "DNN":
        model = DNN(3*256*256, 2)
    elif args.model == "CNN":
        model = CNN()
    elif args.model == "RNN":
        model = RNN()
    else:
        print("error model !")

    # 恢复网络模型
    if args.resume:
        print("=> loading checkpoint: " + args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)
        print("=> checkpoint loaded.")
        print('\n')
    else:
        print("=> 从头开始训练")
        print('\n')

    # 设置 device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), args.lr)

    # 建立损失函数
    criterion = nn.CrossEntropyLoss()

    # 建立 transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))     
    ])

    # 定义数据加载器
    train_dataset = Dataloader(PATH_TRAIN_MINI, transform, args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)

    test_dataset = Dataloader(PATH_TEST_MINI, transform, args)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*4,
                                               shuffle=False, num_workers=args.workers)

    # ---------- step2: 1 Epoch 的训练和测试 ---------- #
    acc_train_list = [] 
    acc_test_list = [] 
    loss_list = []
    for epoch in range(args.epochs):
        # 更新实时的学习率
        adjust_learning_rate(optimizer, epoch, args)

        # 训练网络
        temp_train_acc, temp_loss = train(train_loader, model, criterion, optimizer, epoch, args, device)

        # 测试网络
        temp_test_acc = test(test_loader, model, epoch, args, device)

        print('\n')

        # 更新列表
        acc_train_list.append(temp_train_acc)
        acc_test_list.append(temp_test_acc) 
        loss_list.append(temp_loss)

    # ---------- step3: 绘制曲线 ---------- #
    draw_acc_pic(acc_train_list, acc_test_list, loss_list, args)

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
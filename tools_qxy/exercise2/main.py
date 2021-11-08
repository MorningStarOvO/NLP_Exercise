"""
    本代码用于: NLP 词向量的练习
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 24 日
    具体步骤: 
        step1: 文本数据预处理, 获取 one-hot 表
        step2: 建立网络模型、数据加载器、优化器、损失函数
        step3: 训练词向量
        step4: 词向量测试 
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 
import pprint

# ----- 模型创建相关的 ----- #
import torch 
import torch.optim as optim 
import torch.nn as nn

# ----- 导入自定义的包 ----- #
from option import parse_opt # 导入命令行交互函数
from utils import setup_seed, adjust_learning_rate, txt_preprocess, draw_acc_pic, Svae_Embedding, Test_Embedding # 导入常用函数
from model import NNLM, RNNLM, C_and_W, CBOW, Skip_gram # 导入自定义的模型
from dataset import Dataloader
from process import train

# ----- 数据处理相关的 ----- #
import numpy as np
import json 
from numpy import save 

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ----- 命令行交互相关的 ----- #
import argparse

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle
 
# ----- 创建词云相关的 ----- #
from wordcloud import WordCloud


# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关的 ----- #
TXT_EN = "data/exp2/en.txt"
TXT_ZH = "data/exp2/zh.txt"

# ----- 保存文件相关的 ----- #
ONE_HOT_SAVE = "data/exp2"

WORD_EMBED_SAVE = "output_pic/exp2/word_embedding"

# ==================== 函数实现 ==================== #


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("=> 程序开始运行 !\n")

    # ---------- step0: 输入 args 命令 ---------- #
    args = parse_opt()

    # 设置随机数种子
    setup_seed(args.random)

    # 设置基本参数
    if args.txt_type == "zh":
        path_read = TXT_ZH
        path_save = os.path.join(ONE_HOT_SAVE, "zh_")
    elif args.txt_type == "en":
        path_read = TXT_EN
        path_save = os.path.join(ONE_HOT_SAVE, "en_")
    else:
        print("数据名称 error !")


    # ---------- step1: 文本数据预处理, 获取 one-hot 表 ---------- #
    word_dict, number_dict, n_class, max_len = txt_preprocess(path_read, path_save)

    n_class = n_class + 1 # 这里加 1 是把 EOS 算进去了 

    # ---------- step2: 建立网络模型、数据加载器、优化器、损失函数 ---------- #
    # 创建网络模型
    print("=> creating model: ", args.model)
    print('\n')

    if args.model == "NNLM":
        model = NNLM(n_class, args.embedd_dim, args.n_gram - 1, args)
    elif args.model == "RNNLM":
        model = RNNLM(n_class, args.embedd_dim, args.n_gram - 1, args)
    # elif args.model == "C_W":
    #     model = C_and_W(n_class, args.embedd_dim, args.n_gram - 1, args)
    elif args.model == "CBOW":
        model = CBOW(n_class, args.embedd_dim, args)
    # elif args.model == "Skip-gram":
    #     model = Skip_gram(n_class, args.embedd_dim, n_class, args)

    print(model) 

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

    # 定义数据加载器
    dataset = Dataloader(path_read, max_len, word_dict, args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)

    # ---------- step3: 训练词向量 ---------- #
    acc_list = []
    loss_list = []
    for epoch in range(args.epochs):
        # 更新实时的学习率k
        adjust_learning_rate(optimizer, epoch, args)

        # 训练词向量
        temp_acc, temp_loss = train(loader, model, criterion, optimizer, epoch, args, device, max_len)

        print('\n')

        # 更新列表
        acc_list.append(temp_acc)
        loss_list.append(temp_loss)

    # ----- 绘制曲线 ----- #    
    draw_acc_pic(acc_list, loss_list, args)

    # ----- 保存词向量 ----- #
    Svae_Embedding(model, n_class, args)

    # ---------- step4: 词向量测试 ---------- #
    Test_Embedding(model, n_class, args, word_dict, number_dict)

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
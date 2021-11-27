"""
    本代码用于: XXXX
    创建时间: 2021 年 11 月 09 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 09 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os 
import sys 


# ----- 模型创建相关的 ----- #
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam, SGD#

# ----- 美化输出相关的 ----- #
import pprint 


# ---- 数据处理相关的 ---- # 
import numpy as np 
import pandas as pd 

# ----- 导入自定义的包 ----- #
from utils import build_vocab, build_dict, cal_max_length, Config
from model import NERLSTM 
from dataset import NERdataset

# ----- 设置显卡型号 ----- #
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
def val(config, model):

    # ignore the pad label
    model.eval()

    testset = NERdataset(config.data_dir, 'test', word2id, tag2id, max_length)
    dataloader = DataLoader(testset, batch_size=1) # config.batch_size
    preds, labels = [], []
    for index, data in enumerate(dataloader):
        optimizer.zero_grad()
        corpus, label, length = data
        corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
        score, predict = model(corpus)

        label=label.tolist()
        preds.extend(predict[:length.item()])
        preds.extend(label[0][length.item():])
        labels.extend(label[0])
        # print(predict)
        # print(label[0][len(predict):])
        # print(label[0])

    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    model.train()
    return precision, recall, f1


def train(config, model, dataloader, optimizer):

    # ignore the pad label
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    best_f1 = 0.0
    for epoch in range(config.epoch):
        for index, data in enumerate(dataloader):
            optimizer.zero_grad()
            corpus, label, length = data
            corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()

            # ----- 计算损失 ----- #
            losses = model.neg_log_likelihood(corpus, label.view(-1))
            loss = 0
            for temp in losses:
                loss += temp 
            loss.backward()
            optimizer.step()
            if (index % 200 == 0):
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        prec, rec, f1 = val(config, model)
        if(f1 > best_f1):
            torch.save(model, config.save_model)

# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step0: config() ---------- # 
    config = Config()

    # ---------- step1: 建立网络模型、数据加载器、优化器、损失函数 ---------- # 
    # ----- 建立词向量表 ----- #
    # 需要替换
    word_dict = build_vocab(config.data_dir) 

    # ----- 建立「词典」和「tag 字典」 ----- # 
    word2id, tag2id = build_dict(word_dict)
    
    # ----- 求取最大长度 ----- #
    max_length = cal_max_length(config.data_dir)
    
    # ----- 建立「trainset」 和 「dataloader」 ----- #
    trainset = NERdataset(config.data_dir, 'train', word2id, tag2id, max_length)
    dataloader = DataLoader(trainset, batch_size=config.batch_size)
    print("\n数据加载器已完成 =>")

    # ----- 建立「模型」和「优化器」 ----- #
    nerlstm = NERLSTM(config.embedding_dim, config.hidden_dim, config.dropout, word2id, tag2id).cuda()
    optimizer = Adam(nerlstm.parameters(), config.learning_rate)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nerlstm.to(device)
    
    # ----- 开始训练 ----- #
    print("\n开始训练 =>")
    train(config, nerlstm, dataloader, optimizer)

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
"""
    本代码用于: 记录定义的一些常用函数
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 14 日
    具体的函数:
        setup_seed(): 置随机种子, 使结果可复现 
        adjust_learning_rate(): 动态调整学习率
        txt_preprocess(): 文本数据的预处理, 建立相应的字典
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

# ----- 产生随机数相关的 ----- #
import random
from random import shuffle

# ----- 文件读取相关的 ----- #
import json 

# ----- 创建词云相关的 ----- #
from wordcloud import WordCloud

# ----- 画图相关的 ----- #
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='SimHei.ttf', size=16)    # 创建字体对象

# ----- 降维可视化相关的 ----- #
# from sklearn.manifold import TSNE


# ==================== 设置常量参数 ==================== #
# ----- 存储文件相关的 ----- #
PATH_PIC = "output_pic/exp2"

WORD_EMBED_SAVE = "output_pic/exp2/word_embedding"


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
    if 20 < epoch <= 64: # 20 到 30 轮次学习率降低
        lr = 1e-4
    elif 64 < epoch : # 超过 30 轮次学习率更低
        # lr = 1e-6
        lr = 5e-5

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

        # 建立词云
        str_word_cloud = " ".join(lines)
        word_cloud = WordCloud(background_color='white')
        word_cloud.generate(str_word_cloud)
        word_cloud.to_file(path_save + "WordCloud.png")
        
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


# ----- 绘制曲线 ----- #
def draw_acc_pic(acc_list, loss_list, args):
    """
    绘制曲线
        acc_list: acc 的列表
        loss_list: 损失值的列表
    """
    # 绘制损失值曲线
    plt.figure()
    plt.plot(loss_list, label="loss")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("损失值", fontproperties=font)
    str_title = args.txt_type + "-" + args.model + " 损失值曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()
    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))

    # 汇总训练集和测试集的准确率
    plt.figure()
    plt.plot(acc_list, label="acc")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("准确率", fontproperties=font)
    str_title = args.txt_type + "-" + args.model + " 准确率曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()
    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))


# ----- 保存词向量 ----- #
def Svae_Embedding(model, n_class, args):
    """
    保存词向量
        model: 训练得到的模型
        n_class: 词的总个数
        args: 交互命令
    """
    # 打印模型的参数
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("\n")

    # 定义词向量模型
    embeds = nn.Embedding(n_class, args.embedd_dim)

    # 加载参数
    param_embed = model.state_dict()["lookup.weight"].cpu()
    pretrained_weight = np.array(param_embed)
    embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))

    # 获得词向量
    data_embed = []
    for i in range(n_class):
        temp_embeds = embeds(torch.LongTensor([i])).detach().numpy()
        # print(temp_embeds)
        data_embed.append(temp_embeds)
    data_embed = np.array(data_embed)

    # 存储词向量
    str_path = os.path.join(WORD_EMBED_SAVE, args.txt_type + "-" + args.model + "-WordEmbedding.npy")
    save(str_path, data_embed)


# ----- 测试词向量 ----- #
def Test_Embedding(model, n_class, args, word_dict, number_dict):
    """
    测试词向量
        model: 训练得到的模型
        n_class: 词的总个数
        args: 交互命令
        word_dict: 从 word 到 num
        number_dict: 从 num 到 word
    """

    # 定义词向量模型
    embeds = nn.Embedding(n_class, args.embedd_dim)

    # 加载参数
    param_embed = model.state_dict()["lookup.weight"].cpu()
    pretrained_weight = np.array(param_embed)
    embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))

    # 获得词向量
    data_embed = []
    for i in range(n_class):
        temp_embeds = embeds(torch.LongTensor([i])).detach().numpy()
        # print(temp_embeds)
        data_embed.append(temp_embeds[0])
    data_embed = np.array(data_embed)

    # # t-SNE 可视化
    # tsne = TSNE(n_components=2)
    # tsne.fit_transform(data_embed)
    # data_tsne = tsne.embedding_

    # 找出最近的词
    if args.txt_type == "en":
        for word in["man", "happy", "beautiful"]:
            index = word_dict[word]
            temp_embed = data_embed[index]
            cos_dis = np.array([T.distance.cosine(e, temp_embed) for e in data_embed])
            print(word, " 最相似的词汇有: ")
            print([number_dict[i] for i in cos_dis.argsort()[:10]])
            print("\n")
    elif args.txt_type == "zh":
        for word in["男", "女", "美"]:
            index = word_dict[word]
            temp_embed = data_embed[index]
            cos_dis = np.array([T.distance.cosine(e, temp_embed) for e in data_embed])
            print(word, " 最相似的词汇有: ")
            print([number_dict[i] for i in cos_dis.argsort()[:10]])
            print("\n")
    

    # for word in["", "", ""]:


"""
    本代码用于: 创建数据加载器函数
    创建时间: 2021 年 11 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 26 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- # 
from torch.utils.data import Dataset

# ----- 图像读取相关 ----- #
from PIL import Image

# ----- 模型创建相关 ----- # 
import torch 

# ----- 导入自定义的包 ----- #
from utils import build_glove


# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关 ----- # 


# ----- 存储文件相关 ----- #

START_TAG = "<START>"
STOP_TAG = "<STOP>"

# ==================== 函数实现 ==================== #
# ---------- 定义数据加载器 ---------- #
class NERdataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        
        # self.glove_list = build_glove()
        word2id = build_glove()
        self.corpus = []
        self.label = []
        self.length = []
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            self.length.append(len(corpus_.split()))
            
            if(len(self.corpus[-1]) > max_length):
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
            else:
                while(len(self.corpus[-1]) < max_length):
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag2id['PAD'])
        
        self.corpus = torch.Tensor(self.corpus).float()  
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()

    def __getitem__(self, item):
        # print(self.corpus[item])
        
        return self.corpus[item], self.label[item], self.length[item]

    def __len__(self):
        return len(self.label)
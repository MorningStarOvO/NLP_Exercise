"""
    本代码用于: 建立数据加载器
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 14 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 
from pprint import pprint 

# ----- 数据处理相关的包 ----- #
import numpy as np 

# ----- 模型创建相关的 ----- # 
import torch 
from torch.utils.data import Dataset

# ==================== 函数实现 ==================== #
# ---------- 定义数据加载器 ---------- #
class Dataloader(Dataset):
    def __init__(self, path, max_len, word_dict, args=None):
        """
        path: 数据集总路径
        max_len: 最大长度
        args: 命令行交互内容
        """
        # 获取基本信息
        self.max_len = max_len
        self.path = path 
        self.args = args 
        self.word_dict = word_dict
        self.ctxt_win = args.ctxt_win
        self.neg_size = args.neg_size
        
        # 获得 n-gram 的步长
        step = self.args.n_gram - 1
        # 得到 EOS 的 index
        index_EOS = len(self.word_dict)

        # 处理文本数据
        self.txt_list = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for temp_line in lines:
                temp = temp_line.split()
                self.txt_list.append(temp)

        # 处理各个模型的数据加载器 
        self.label_list = []
        self.data_list = []
        self.center_list = [] # 中心词
        self.win_list = [] # 除中心词外 win 的词
        if self.args.model == "NNLM":
            for temp in self.txt_list: 
                for i in range(len(temp)- step):
                    temp_data = []
                    for j in range(step):
                        temp_data.append(self.word_dict[temp[i+j]])
                    self.data_list.append(temp_data)

                    temp_label = self.word_dict[temp[i+step]]
                    self.label_list.append(temp_label)
        elif self.args.model == "RNNLM":
            for temp in self.txt_list:
                temp_data = []
                temp_label = []
                for i in range(len(temp)):
                    temp_data.append(self.word_dict[temp[i]])
                    if i > step:
                        temp_label.append(self.word_dict[temp[i]])
                for j in range(len(temp), max_len):
                    temp_data.append(index_EOS)
                    temp_label.append(index_EOS)
                self.data_list.append(temp_data)
                self.label_list.append(temp_label)
        elif args.model == "CBOW":
            for temp in self.txt_list:
                for i in range(self.ctxt_win, len(temp) - self.ctxt_win):
                    temp_ceter_word = self.word_dict[temp[i]]
                    temp_win_word = []
                    for j in range(self.ctxt_win):
                        temp_win_word.append(self.word_dict[temp[i-self.ctxt_win+j]])
                    for j in range(self.ctxt_win):
                        temp_win_word.append(self.word_dict[temp[i+j+1]])
                    
                    self.win_list.append(temp_win_word)
                    self.center_list.append(temp_ceter_word)


    def __getitem__(self, index):

        if self.args.model == "NNLM":
            # 获得当前的 data 和 label
            data = np.array(self.data_list[index])
            data = torch.from_numpy(data)
            label = self.label_list[index] 
            # print("label: ", label)
        elif self.args.model == "RNNLM":
            # 获得当前的 data 和 label
            data = np.array(self.data_list[index])
            data = torch.from_numpy(data)
            label = np.array(self.label_list[index])
            label = torch.from_numpy(label)
            
        # elif self.args.model == "C_W":

        elif self.args.model == "CBOW": 
            data = np.array(self.win_list[index])
            data = torch.from_numpy(data)
            label = np.array(self.center_list[index])
            label = torch.from_numpy(label)
        # elif self.args.model == "Skip-gram":
 
        return data, label

    def __len__(self):
        return len(self.txt_list)

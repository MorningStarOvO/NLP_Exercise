"""
    本代码用于: 建立数据加载器
    创建时间: 2021 年 12 月 26 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 
from pprint import pprint 

# ----- 数据处理相关的包 ----- #
import numpy as np 

# ----- 模型创建相关的 ----- # 
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

# ----- 日志处理相关的 ----- #
import logging

# ==================== 函数实现 ==================== #
# ----- 加载「中文的」tokenize ----- #
def chinese_tokenizer_load():
    sp_ch = spm.SentencePieceProcessor()
    sp_ch.Load('./data/ch.model')
    logging.info(f'SentencePiece loaded at ./data/ch.model')
    return sp_ch

# ----- 加载「英文的」tokenize ----- #
def english_tokenizer_load():
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load('./data/en.model')
    logging.info(f'SentencePiece loaded at ./data/en.model')
    return sp_en

# ----- 生成下三角矩阵 ----- #
# 用于 Mask 操作
def subsequent_mask(size): 
    """Mask out subsequent positions."""
    # 设定 shape
    attn_shape = (1, size, size)

    # 右上角(不含主对角线)为全 1，左下角(含主对角线)为全 0 的矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    # 返回一个右上角(不含主对角线)为全 False，左下角(含主对角线)为全 True 矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    # ---- 掩码操作 ---- #
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, ch_data_path, en_data_path, rank=None):
        self.en_sent, self.cn_sent = self.get_dataset(ch_data_path, en_data_path, sort=True) 
        self.sp_en = english_tokenizer_load()
        self.sp_ch = chinese_tokenizer_load()
        self.PAD = self.sp_en.pad_id()  # 0
        self.BOS = self.sp_en.bos_id()  # 2
        self.EOS = self.sp_en.eos_id()  # 3
        self.rank = rank

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, ch_data_path, en_data_path, sort=False):
        with open(ch_data_path, 'r') as f:
            cn_sent = f.readlines()
        with open(en_data_path, 'r') as f:
            en_sent = f.readlines()
        assert len(cn_sent) == len(en_sent), f"Number of lines in {ch_data_path} must be equal to {en_data_path}"
        if sort:
            sorted_index = self.len_argsort(en_sent)
            en_sent = [en_sent[i] for i in sorted_index]
            cn_sent = [cn_sent[i] for i in sorted_index]
        return en_sent, cn_sent

    def __getitem__(self, idx):
        en_text = self.en_sent[idx]
        cn_text = self.cn_sent[idx]
        return [en_text, cn_text]

    def __len__(self):
        return len(self.en_sent)

    def collate_fn(self, batch):
        # ----- 读取句子 ----- #
        src_text = [x[0] for x in batch]    # en_text
        tgt_text = [x[1] for x in batch]    # cn_text

        # ----- 为句子加上起止符 ----- #
        # Tokenize with SentencePiece, add [BOS] & [EOS]
        src_tokens = [[self.BOS] + self.sp_en.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_ch.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # ----- 批次内进行 padding 操作 ----- #
        # Pad Sequence
        src_pad = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        tgt_pad = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)
        if self.rank != None:
            src_pad = src_pad.to(self.rank)
            tgt_pad = tgt_pad.to(self.rank)

        return Batch(src_text, tgt_text, src_pad, tgt_pad, self.PAD)
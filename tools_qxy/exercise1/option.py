"""
    本代码用于: 实现 argparse 的设置
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""

import argparse
from pprint import pprint

def parse_opt():
    parser = argparse.ArgumentParser(description='NLP Homework1 !')

    # 选择模型
    parser.add_argument('--model', default="RNN", type=str,
                        help="Name of model: DNN、CNN、RNN")
  
    # 设置随机数种子
    parser.add_argument('--random', default=1234, type=int, 
                        help="random seed")

    # 设置 CPU 的 workers 
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    # 设置批次
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    # 设置学习率
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')

    # 设置恢复模型的路径
    parser.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # 设置训练的 epoch 数目
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    pprint("parser 的输入参数: ")
    pprint(vars(args))
    print('\n')

    return args 
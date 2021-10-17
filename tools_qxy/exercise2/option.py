"""
    本代码用于: 定义命令行交互的内容
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 14 日
"""

# ==================== 导入必要的包 ==================== #
import argparse
from pprint import pprint

# ==================== 定义 argparse 函数 ==================== #
def parse_opt():
    parser = argparse.ArgumentParser(description='NLP Homework2 !')

    # 选择词向量模型
    # RNNLM 暂未实现
    parser.add_argument('--model', default="NNLM", type=str,
                        help="Name of model: NNLM、RNNLM、C&W、CBOW、Skip-gram")
  
    # 选择中英文本
    parser.add_argument('--txt_type', default="en", type=str,
                        help="Name of txt type: en、zh")

    # 设置词向量维度
    parser.add_argument('--embedd_dim', default=128, type=int, 
                        help="embedding dim")

    # 设置 n-gram 的 n 
    parser.add_argument('--n_gram', default=2, type=int, 
                        help="n-gram")

    # 设置随机数种子
    parser.add_argument('--random', default=4321, type=int, 
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
    parser.add_argument('--epochs', default=256, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    pprint("parser 的输入参数: ")
    pprint(vars(args))
    print('\n')

    return args 
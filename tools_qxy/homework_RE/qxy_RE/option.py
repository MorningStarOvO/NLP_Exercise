"""
    本代码用于: 定义命令行交互的内容
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""

# ==================== 导入必要的包 ==================== #
import argparse
from pprint import pprint

# ==================== 定义 argparse 函数 ==================== #
def parse_opt():
    parser = argparse.ArgumentParser(description='NLP Homework2 !')

    parser.add_argument('--model', default="CNN", type=str,
                        help="Name of model: CNN、LSTM_Att、LSTM_MaxPool")
    parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")
    parser.add_argument('--embedding_file', default='data/embeddings/vector_50d.txt', help="Path to embeddings file.")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--gpu', default=-1, help="GPU device number, 0 by default, -1 means CPU.")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before training")


    args = parser.parse_args()
    pprint("parser 的输入参数: ")
    pprint(vars(args))
    print('\n')

    return args 
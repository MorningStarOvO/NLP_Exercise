"""
    本代码用于: 定义命令行交互的内容
    创建时间: 2021 年 12 月 28 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""

# ==================== 导入必要的包 ==================== #
import argparse
from pprint import pprint

# ==================== 定义 argparse 函数 ==================== #
def parse_opt():
    parser = argparse.ArgumentParser(description='NLP Homework Machine Translation !')

    # ---------- 设置「数据路径」相关的参数 ---------- #
    parser.add_argument("--data_dir", type=str, default='./data', \
                        help="Path to sentence piece model & vocab dir")
    parser.add_argument("--train_ch_data_path", type=str, default='./data/corpus/train.zh', \
                        help="training data file path")
    parser.add_argument("--train_en_data_path", type=str, default='./data/corpus/train.en', \
                        help="training data file path")
    parser.add_argument("--test_ch_data_path", type=str, default='./data/corpus/test.zh', \
                        help="test data file path")
    parser.add_argument("--test_en_data_path", type=str, default='./data/corpus/test.en', \
                        help="test data file path")
    parser.add_argument("--dev_ch_data_path", type=str, default='./data/corpus/valid.zh', \
                        help="dev data file path")
    parser.add_argument("--dev_en_data_path", type=str, default='./data/corpus/valid.en', \
                        help="dev data file path")

    # ---------- 设置「模型」和「log」路径相关的参数 ---------- #
    parser.add_argument("--model_path", type=str, default='./output/model.pth', \
                        help="model save path")
    parser.add_argument("--model_path_best", type=str, default='./output/model_best.pth', \
                        help="best model save path")
    parser.add_argument("--temp_dir", type=str, default='./output/temp', \
                        help="temp data dir path")
    parser.add_argument("--log_path", type=str, default='./output/train_continue.log', \
                        help="log save path")
    parser.add_argument("--output_path", type=str, default='./output/output_continue.txt', \
                        help="test predict file path")


    # ---------- 训练超参数相关的 ---------- #
    parser.add_argument("--batch_size", type=int, default=16, help="Dataloader batch size")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--continue_training", type=bool, default=True, \
                        help="Whether to continue Training")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, \
                        help="Warmup proportion")
    parser.add_argument("--beam_size", type=int, default=3, \
                        help="beam size for decode")
    parser.add_argument("--max_len", type=int, default=60, \
                        help="max_len for decode")
    parser.add_argument("--n_gpu", type=int, default=1, help='Number of GPUs in one node')
    parser.add_argument("--n_node", type=int, default=1, help='Number of nodes in total')
    parser.add_argument("--node_rank", type=int, default=0, help='Node rank for this machine. 0 for master, and 1,2... for slaves')

    # ---------- 模型参数相关的 ---------- #
    parser.add_argument("--d_model", type=int, default=512, help="Input & Output dimension for Translation Model")
    parser.add_argument("--n_heads", type=int, default=8, help="Multi-head Num")
    parser.add_argument("--n_layers", type=int, default=6, help="Encoder Layer Number & Decoder Layer Number")
    parser.add_argument("--d_k", type=int, default=64, help="Dimension of K")
    parser.add_argument("--d_v", type=int, default=64, help="Dimension of V")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of Position-wise Feed-Forward Inner Layer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for SublayerConnection, which contain layer norm + dropout")
    parser.add_argument("--padding_idx", type=int, default=0, help="Padding Index in Vocab")
    parser.add_argument("--bos_idx", type=int, default=2, help="[BOS] Index in Vocab")
    parser.add_argument("--eos_idx", type=int, default=3, help="[EOS] Index in Vocab")
    parser.add_argument("--src_vocab_size", type=int, default=32000, help="Source language (English) vocab size")
    parser.add_argument("--tgt_vocab_size", type=int, default=32000, help="Target language (Chinese) vocab size")
    
    # ----- 输出参数 ----- #
    args = parser.parse_args()
    pprint("parser 的输入参数: ")
    pprint(vars(args))
    print('\n')

    return args 
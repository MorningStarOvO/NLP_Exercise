"""
    本代码用于: 用「transformer」实现「机器翻译」
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import time
import os

# ----- 命令行交互相关 ----- #
import logging

# ----- 训练模型相关的包 ----- # 
import torch 
from torch.utils.data import DataLoader
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

# ----- 自定义的包 ----- #
from option import parse_opt
import utils
from process import train, test
from dataset import MTDataset
from model import transformer_model as transformer
# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
def run_train(rank, args):
    utils.set_logger(args.log_path)
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    train_dataset = MTDataset(ch_data_path=args.train_ch_data_path, en_data_path=args.train_en_data_path, rank=rank)
    dev_dataset = MTDataset(ch_data_path=args.dev_ch_data_path, en_data_path=args.dev_en_data_path, rank=rank)
    logging.info(f"-------- Dataset Build! --------")


    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    logging.info("-------- Get Dataloader! --------")

    model = transformer(args.src_vocab_size, args.tgt_vocab_size, args.n_layers,
                       args.d_model, args.d_ff, args.n_heads, args.dropout)
    if torch.cuda.is_available():    
        model.cuda(rank)

    total_steps = 1.0 * len(train_dataloader) * args.epoch_num
    warmup_steps = args.warmup_proportion * total_steps
    logging.info(f"Scheduler: total_steps:{total_steps}, warmup_steps:{warmup_steps}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, rank, args)


def run_test(rank, args):
    utils.set_logger(args.log_path)
    if not os.path.exists(args.temp_dir):
        os.makedirs(args.temp_dir)

    test_dataset = MTDataset(ch_data_path=args.test_ch_data_path, en_data_path=args.test_en_data_path, rank=rank) 
    logging.info(f"-------- Dataset Build! --------")

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn)
    logging.info("-------- Get Dataloader! --------")

    # 初始化模型
    model = transformer(args.src_vocab_size, args.tgt_vocab_size, args.n_layers,
                        args.d_model, args.d_ff, args.n_heads, args.dropout)
    if torch.cuda.is_available():    # Move model to GPU:rank
        model.cuda(rank)

    test(test_dataloader, model, rank, args)


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step0: 进行命令行交互 ---------- #
    args = parse_opt()
    utils.setup_seed(1)

    # ---------- step1: 训练模型 ---------- #
    run_train(0, args)

    # ---------- step2: 测试模型 ---------- # 
    run_test(0, args)


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 !")
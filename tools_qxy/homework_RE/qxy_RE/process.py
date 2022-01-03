"""
    本代码用于: 定义训练过程和测试过程
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- #
import torch

# ----- 可视化运行过程 ----- #
from tqdm import tqdm
from tqdm import trange

# ----- 导入命令行交互的包 ----- # 
import logging 

# ----- 导入创建模型相关的包 ----- # 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_recall_fscore_support

# ----- 导入自定义的包 ----- # 
import utils 

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
# ---------- 训练过程的实现 ---------- #
def train(model, data_iterator, optimizer, scheduler, params, steps_num):
    """Train the model on `steps_num` batches"""
    # set model to training mode
    model.train()
    # scheduler.step()
    # a running average object for loss
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    t = trange(steps_num)
    for _ in t:
        # fetch the next training batch
        batch_data, batch_labels = next(data_iterator)

        # compute model output and loss
        batch_output = model(batch_data)
        loss = model.loss(batch_output, batch_labels)

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        # optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    return loss_avg()

# ---------- 测试过程的实现 ---------- # 
def evaluate(model, data_iterator, num_steps, metric_labels):
    """Evaluate the model on `num_steps` batches."""
    # set model to evaluation mode
    model.eval()

    output_labels = list()
    target_labels = list()

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
        batch_data, batch_labels = next(data_iterator)
        
        # compute model output
        batch_output = model(batch_data)  # batch_size x num_labels
        batch_output_labels = torch.max(batch_output, dim=1)[1]
        output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
        target_labels.extend(batch_labels.data.cpu().numpy().tolist())

    # Calculate precision, recall and F1 for all relation categories
    p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average='micro')
    p_r_f1 = {'precison': p_r_f1_s[0] * 100,
              'recall': p_r_f1_s[1] * 100,
              'f1': p_r_f1_s[2] * 100}
    return p_r_f1

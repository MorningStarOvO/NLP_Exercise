"""
    本代码用于: 定义训练过程和测试过程
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- #
import torch

# ----- 可视化运行过程 ----- #
from tqdm import tqdm

# ==================== 设置常量参数 ==================== #
# ----- 读取文件相关 ----- # 
PATH_TRAIN_MINI = "data/exp1_train"
PATH_TEST_MINI = "data/exp1_val"

# ----- 存储文件相关 ----- #
PATH_CHECKPOINT = "checkpoint/exp1" 
PATH_LOGS = "logs/exp1"

# ==================== 函数实现 ==================== #
# ---------- 定义训练网络 ---------- #
def train(train_loader, model, criterion1, optimizer, epoch, args, device):
    """
    训练网络相关的
        train_loader: 训练集的数据加载器
        model: 要训练的模型
        criterion1: 损失函数
        optimizer: 优化器
        epoch: 当前的轮数
        args: 交互命令
    """

    # 模式切换为 train
    model.train()

    running_loss = 0.0
    correct_sum = 0
    pic_sum = 0

    count = 0

    # 读取数据
    # for i, (images, target) in enumerate(tqdm(train_loader)):
    for i, (images, target) in enumerate(train_loader):
        # 将数据加载到 GPU 上
        images = images.to(device)
        target = target.to(device)

        # 正向传播
        output = model(images)

        # 计算损失
        loss = criterion1(output, target)

        # compute gradient and do optimizer step
        optimizer.zero_grad() # 梯度置 0
        loss.backward() # 反向传播
        optimizer.step() # 优化

        # loss.item() 可直接获得 loss 的数值
        # running_loss 为不断累加 loss 的值
        running_loss += loss.item()

        correct = output.detach().argmax(1) == target
        correct_sum += torch.sum(correct)
        pic_sum += len(target)

        count += 1

    # 输出信息
    acc = round(float(correct_sum) / pic_sum, 4)
    print("Epoch " + str(epoch) +  " 的训练集准确率为: ", acc)
    print("Epoch " + str(epoch) +  " 的训练集损失值为: ", running_loss/count)

    # 保存模型
    path_str = os.path.join(PATH_CHECKPOINT, args.model)
    path_str = os.path.join(path_str, str(acc) + "_" + str(epoch) + ".pth.tar")
    torch.save(model.state_dict(), path_str)

    return acc, running_loss/count

# ---------- 定义测试网络 ---------- #
def test(test_loader, model, epoch, args, device):
    """
    测试网络相关的: 
        test_loader: 测试集的数据加载器
        model: 测试的模型
        epoch: 当前轮数
        args: 命令行交互
    """

    # 转换模式
    model.eval()

    correct_sum = 0
    pic_sum = 0

    # 读取数据
    # for i, (images, target) in enumerate(tqdm(test_loader)):  
    for i, (images, target) in enumerate(test_loader):  
        # 将数据加载到 GPU 上
        images = images.to(device)
        target = target.to(device)

        # 正向传播
        output = model(images)

        # 计算正确的个数 
        correct = output.detach().argmax(1) == target
        correct_sum += torch.sum(correct)
        pic_sum += len(target)

    # 输出信息
    acc = round(float(correct_sum) / pic_sum, 4)
    print("Epoch " + str(epoch) +  " 的测试集准确率为: ", acc)

    return acc 
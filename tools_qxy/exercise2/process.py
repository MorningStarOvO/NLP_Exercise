"""
    本代码用于: 定义训练过程和测试过程
    创建时间: 2021 年 10 月 14 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 14 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- #
import torch

# ----- 可视化运行过程 ----- #
from tqdm import tqdm


# ==================== 设置常量参数 ==================== #
# ----- 存储地址相关 ----- # 
PATH_CHECKPOINT = "checkpoint/exp2"


# ==================== 函数实现 ==================== #
# ---------- 定义训练网络 ---------- #
def train(train_loader, model, criterion1, optimizer, epoch, args, device, max_len):
    """
    训练网络相关的
        train_loader: 训练集的数据加载器
        model: 要训练的模型
        criterion1: 损失函数
        optimizer: 优化器
        epoch: 当前的轮数
        args: 交互命令
        max_len: 字符串最大长度
    """
    # 模式切换为 train
    model.train()

    running_loss = 0.0
    correct_sum = 0

    count = 0
    
    for i, (data, target) in enumerate(tqdm(train_loader)):
        # print(data.shape)
        # print(target.shape)
        # 将数据加载到 GPU 上
        data = data.to(device)
        target = target.to(device)
        # if args.model == "NNLM":
        #     target = target.squeeze(1)

        # 正向传播
        output = model(data)

        # print(target.size())
        # print(output.size())

        # 计算损失
        loss = 0
        if args.model == "NNLM" or args.model == "CBOW":
            loss = criterion1(output, target)
        elif args.model == "RNNLM":
            for i in range(target.size()[1]):
                loss += criterion1(output[:, i, :], target[:,i])

        optimizer.zero_grad() # 梯度置 0
        loss.backward() # 反向传播
        optimizer.step() # 优化

        # loss.item() 可直接获得 loss 的数值
        # running_loss 为不断累加 loss 的值
        running_loss += loss.item()

        if args.model == "NNLM" or args.model == "CBOW":
            correct = output.detach().argmax(1) == target
            correct_sum += torch.sum(correct)
            count += len(target)
        elif args.model == "RNNLM":
            for i in range(target.size()[1]):
                correct = output[:, i, :].detach().argmax(1) == target[:, i]
                correct_sum += torch.sum(correct)

                count += len(target[:, i])
                # print(len(target[:, i]))


    # 输出信息
    acc = round(float(correct_sum) / count, 4)
    print("\n")
    print("Epoch " + str(epoch) +  " 的训练集准确率为: ", acc)
    print("Epoch " + str(epoch) +  " 的训练集损失值为: ", running_loss/count)
    print("\n")

    # 保存模型
    if epoch % 16 == 0:
        path_str = os.path.join(PATH_CHECKPOINT, args.model)
        path_str = os.path.join(path_str, str(acc) + "_" + str(epoch) + ".pth.tar")
        torch.save(model.state_dict(), path_str)

    return acc, running_loss/count

"""
    本代码用于: 用于定义 DNN、CNN、RNN 模型
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 模型创建相关的 ----- #
import torch
import torch.nn.functional as F  
import torch.nn as nn

# ==================== 函数实现 ==================== #
# ---------- 定义 DNN 网络 ---------- #
class DNN(nn.Module): 
    def __init__(self, n_feature, n_output):
        # 继承 __init__ 功能
        super(DNN, self).__init__()

        self.f1 = torch.nn.Linear(n_feature, 2048)
        self.f2 = torch.nn.Linear(2048, 1024)
        self.f3 = torch.nn.Linear(1024, 512)
        self.f4 = torch.nn.Linear(512, 256)
        self.f5 = torch.nn.Linear(256, 128)
        self.f6 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, n_output)  

    def forward(self, x):
        # 获得 x 的维度
        B, C, H, W = x.shape

        # 改变 x 的维度
        x = x.view(B, -1)

        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.f1(x)) 
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))
        x = F.relu(self.f5(x))
        x = F.relu(self.f6(x))

        x = self.out(x)                
        return x

# ---------- 定义 CNN 网络 ---------- #
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # input shape (3, 256, 256)
        self.conv1 = nn.Sequential(  
            nn.Conv2d(
                in_channels=3,      
                out_channels=16,    
                kernel_size=5,      
                stride=1,           
                padding=2,      
            ),      
            nn.BatchNorm2d(16),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 128, 128)
        )

        # input shape (16, 128, 128)
        self.conv2 = nn.Sequential(  
            nn.Conv2d(16, 32, 5, 1, 2),  
            nn.BatchNorm2d(32),
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 输出形状 (32, 64, 64)
        )

        # input shape (32, 64, 64)
        self.conv3 = nn.Sequential(  
            nn.Conv2d(32, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 输出形状 (64, 32, 32)
        )

        # input shape (64, 32, 32)
        self.conv4 = nn.Sequential(  
            nn.Conv2d(64, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 输出形状 (128, 16, 16)
        )

        # input shape (128, 16, 16)
        self.conv5 = nn.Sequential(  
            nn.Conv2d(128, 256, 3, 1, 1),  
            nn.BatchNorm2d(256),
            nn.ReLU(),  
            nn.MaxPool2d(2),  # 输出形状 (256, 8, 8)
        )

        # 全连接层
        self.f1 = torch.nn.Linear(256 * 8 * 8, 2048)
        self.f2 = torch.nn.Linear(2048, 1024)
        self.out = nn.Linear(1024, 2)

    def forward(self, x):
        # 卷积操作
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # 全连接操作
        x = x.view(x.size(0), -1)   
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        output = self.out(x)
        return output


# ---------- 定义 RNN 网络 ---------- #
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(     
            input_size=256,      # 图片每行的数据像素点
            hidden_size=128,
            num_layers=2,       
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, 2)    # 输出层

    def forward(self, x):
        # 获得 x 的维度
        B, C, H, W = x.shape

        # 改变 x 的维度
        x = x.view(B, -1, W) 

        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out
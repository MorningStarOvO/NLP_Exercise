"""
    本代码用于: 用于定义 NNLM、RNNLM、C&W、CBOW、Skip-gram
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
# ---------- 定义 NNLM 模型 ---------- #
class NNLM(nn.Module):
    def __init__(self, n_class, dim_look_up, n_step, args):
        super(NNLM, self).__init__()

        # 定义基本参数
        self.n_hidden = 256
        self.n_class = n_class 
        self.dim_look_up = dim_look_up
        self.n_step = n_step

        # 这里定义的模型将 W 去除了，未有从 look-up 到最后的线
        self.lookup = nn.Embedding(n_class, dim_look_up)
        self.H = nn.Linear(n_step * dim_look_up, self.n_hidden)
        self.U = nn.Linear(self.n_hidden, n_class)

    def forward(self, x):
        x = self.lookup(x)
        x = x.view(-1, self.n_step * self.dim_look_up)

        x = torch.tanh(self.H(x))
        output = self.U(x)

        return output 

# ---------- 定义 RNNLM 模型 ---------- #
class RNNLM(nn.Module):
    def __init__(self, n_class, dim_look_up, n_step, args):
        super(RNNLM, self).__init__()

        # 定义基本参数
        self.n_hidden = 256
        self.n_class = n_class 
        self.dim_look_up = dim_look_up
        self.n_step = n_step

        # 定义模型结构
        self.lookup = nn.Embedding(self.n_class, self.dim_look_up)
        self.rnn = nn.LSTM(
            input_size=self.dim_look_up,
            hidden_size=self.n_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(self.n_hidden, self.n_class)

    def forward(self, x):
        B = x.shape[0]
        x = self.lookup(x)
        x = x.view(B, -1, self.dim_look_up)

        r_out, (h_n, h_c) = self.rnn(x, None)
        output = self.output(r_out[:, -1, :])

        return output

# ---------- 定义 C&W 模型 ---------- #
class C_and_W(nn.Module):
    def __init__(self, n_class, dim_look_up, n_step, args):
        super(C_and_W, self).__init__()

        # 定义基本参数
        self.n_hidden = 512
        self.n_class = n_class 
        self.dim_look_up = dim_look_up
        self.n_step = n_step

        # 定义模型结构
        self.lookup = nn.Embedding(n_class, dim_look_up)
        self.H = nn.Linear(n_step * dim_look_up, self.n_hidden)
        self.U = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        x = self.lookup(x)
        x = x.view(-1, self.n_step * self.dim_look_up)

        x = torch.tanh(self.H(x))
        output = self.U(x)

        return output 


# ---------- 定义 CBOW 模型 ---------- #
class CBOW(nn.Module):
    def __init__(self, n_class, dim_look_up, n_step, args):
        super(CBOW, self).__init__()

        # 定义基本参数
        self.n_class = n_class 
        self.dim_look_up = dim_look_up
        self.n_step = n_step

        # 定义模型结构
        self.lookup = nn.Embedding(n_class, dim_look_up)
        self.output = nn.Linear(n_step * dim_look_up, self.n_class)

    def forward(self, x):
        x = self.lookup(x)
        x = x.view(-1, self.n_step * self.dim_look_up)

        output = self.output(x)

        return output 

# ---------- 定义 Skip-gram 模型 ---------- #
class Skip_gram(nn.Module):
    def __init__(self, n_class, dim_look_up, n_class_out,args):
        super(Skip_gram, self).__init__()

        # 定义基本参数
        self.n_class = n_class 
        self.dim_look_up = dim_look_up

        # 定义模型结构
        self.lookup = nn.Embedding(n_class, dim_look_up)
        self.output = nn.Linear(dim_look_up, n_class_out)

    def forward(self, x):
        x = self.lookup(x)
        x = x.view(-1, self.dim_look_up)

        output = self.output(x)

        return output 
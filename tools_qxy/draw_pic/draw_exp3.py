"""
    本代码用于: XXXX
    创建时间: 2021 年 11 月 30 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 11 月 30 日
"""
# ==================== 导入必要的包 ==================== #
import time
import os 


# ----- 数据处理相关的 ----- #
import pandas as pd

# ----- 画图相关的 ----- #
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='SimHei.ttf', size=16)    # 创建字体对象

from prettytable import PrettyTable

# ==================== 设置常量参数 ==================== #
init_f1_list = []
init_score0_f1_list = []
crf_f1_list = []
crf_score0_f1_list = []


PATH_PIC="output_pic/exp3"
# ==================== 函数实现 ==================== #


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ----- 建立表格 ----- #
    table = PrettyTable(["", "LSTM", "CRF+LSTM"])
    table.add_row(["0 F1",  0.99, 0.99 ])
    table.add_row(["1 F1",  0.83, 0.79 ])
    table.add_row(["2 F1",  0.87, 0.86 ])
    table.add_row(["3 F1",  0.78, 0.79 ])
    table.add_row(["4 F1",  0.84, 0.87 ])
    table.add_row(["5 F1",  0.87, 0.88 ])
    table.add_row(["6 F1",  0.85, 0.86 ])
    table.add_row(["accuracy", 0.97, 0.97])
    print(table) 

    # ----- 绘制图片 ----- #
    with open("logs/exp3/crf_score0.log", 'r') as f:
        lines = f.readlines()
        acc_list = []
        for line in lines:
            if "accuracy" in line:
                acc = line.split()[1]
                acc_list.append(acc) 

    with open("logs/exp3/initial_score0.log", 'r') as f:
        lines = f.readlines()
        acc_list_init = []
        for line in lines:
            if "accuracy" in line:
                acc = line.split()[1]
                acc_list_init.append(acc) 

    # 绘制损失值曲线
    plt.figure()
    plt.plot(acc_list, label="CRF+LSTM") 
    plt.plot(acc_list_init, label="LSTM")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("ACC", fontproperties=font)
    str_title = "ACC 训练曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()
    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))




    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
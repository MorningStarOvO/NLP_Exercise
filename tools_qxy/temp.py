"""
    本代码用于: 随便写写测试
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关 ----- #
import time
import os

# ----- 数据处理相关 ----- #
import numpy as np
import torch 

# ==================== 设置常量参数 ==================== #

# ==================== main 函数 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    word_2_embed = {}
    with open("data/glove/glove.6B.300d.txt", encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]
            temp = np.array(line[1:], dtype=np.float32)
            word_2_embed[word] = temp 

        print(torch.Tensor(word_2_embed["unk"]).double())


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")

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

# ----- 画图相关 ----- #
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='SimHei.ttf', size=16)    # 创建字体对象

# ==================== 设置常量参数 ==================== #
PATH_CNN_ACC = "output_pic/exp1/CNN 测试集准确率.npy"
PATH_DNN_ACC = "output_pic/exp1/DNN 测试集准确率.npy"
PATH_RNN_ACC = "output_pic/exp1/RNN 测试集准确率.npy"

PATH_PIC = "output_pic/exp1"
# ==================== main 函数 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ----- 绘制 3 条曲线 ----- #
    acc_CNN = np.load(PATH_CNN_ACC)
    acc_DNN = np.load(PATH_DNN_ACC)
    acc_RNN = np.load(PATH_RNN_ACC)

    # 汇总训练集和测试集的准确率
    plt.figure()
    plt.plot(acc_CNN, 'o-', label="CNN acc")
    plt.plot(acc_DNN, 'o-', label="DNN acc")
    plt.plot(acc_RNN, 'o-', label="RNN acc")
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("准确率", fontproperties=font)
    str_title = "CNN、DNN、RNN 的测试集准确率曲线"
    plt.title(str_title, fontproperties=font)
    plt.legend()

    # 转换为列表
    # print(type(acc_CNN))
    acc_CNN = acc_CNN.tolist()
    acc_DNN = acc_DNN.tolist()
    acc_RNN = acc_RNN.tolist()

    # 标出最高准确率的位置
    x_max = acc_CNN.index(max(acc_CNN))
    y_max = round(max(acc_CNN), 4)
    # 画点
    plt.scatter(x_max,y_max,s=100,color='b')
    # 标注文本
    str_annotate = str(y_max)
    plt.text(x_max-1, y_max, str_annotate,fontdict={'size':'16','color':'b'})

    # 标出最高准确率的位置
    x_max = acc_DNN.index(max(acc_DNN))
    y_max = round(max(acc_DNN), 4)
    # 画点
    plt.scatter(x_max,y_max,s=100,color='b')
    # 标注文本
    str_annotate = str(y_max)
    plt.text(x_max-1, y_max, str_annotate,fontdict={'size':'16','color':'b'})

    # 标出最高准确率的位置
    x_max = acc_RNN.index(max(acc_RNN))
    y_max = round(max(acc_RNN), 4)
    # 画点
    plt.scatter(x_max,y_max,s=100,color='b')
    # 标注文本
    str_annotate = str(y_max)
    plt.text(x_max-1, y_max, str_annotate,fontdict={'size':'16','color':'b'})


    plt.savefig(os.path.join(PATH_PIC, str_title+".jpg"))


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 ！")
"""
    本代码用于: 
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
import time


# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")


    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 !")
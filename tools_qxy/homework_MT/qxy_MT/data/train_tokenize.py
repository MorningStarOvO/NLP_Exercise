"""
    本代码用于: 训练 tokenize
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 导入系统操作相关的包 ----- #
import time
import os

# ----- tokenize 训练相关 ----- #
import sentencepiece as spm

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
def train(input_file, vocab_size, model_prefix, model_type, character_coverage):
    
    # 修改路径到 . 下
    os.chdir(os.path.abspath('.'))

    # 设置训练参数
    args =  f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} '\
            f'--character_coverage={character_coverage} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'

    # 训练模型
    spm.SentencePieceTrainer.Train(args)

# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step1: 训练 en 的 tokenize ---------- # 
    en_input = './corpus/test.en,./corpus/train.en,./corpus/valid.en'
    en_vocab_size = 32000
    en_model_prefix = 'en'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_prefix, en_model_type, en_character_coverage)


    # ---------- step2: 训练 ch 的 tokenize ---------- #
    ch_input = './corpus/test.zh,./corpus/train.zh,./corpus/valid.zh'
    ch_vocab_size = 32000
    ch_model_prefix = 'ch'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_prefix, ch_model_type, ch_character_coverage)



    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 !")
"""
    本代码用于: 建立词表
    创建时间: 2022 年 1 月 1 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 1 月 1 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 导入系统操作相关的 ----- #
import time
import os 

# ----- 导入命令行交互的 ----- # 
import argparse 
from pprint import pprint

# ----- 导入数据处理的 ----- #
from collections import Counter
import json

# ==================== 设置常量参数 ==================== #


# ==================== 函数实现 ==================== #
# ----- 保存 TXT ----- #
def save_to_txt(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, 'w') as f:
        for token in vocab:
            f.write(token + '\n')
            
# ----- 保存到 JSON ----- #
def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

# ----- 更新词汇表 ----- #
def update_vocab(txt_path, vocab):
    """Update word and label vocabulary from dataset"""
    size = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.endswith('...'):
                line = line.rstrip('...')
            word_seq = line.split('\t')[-1].split(' ')
            vocab.update(word_seq)
            size = i
    return size + 1

# ----- 更新 label ----- #
def update_labels(txt_path, labels):
    """Update label vocabulary from dataset"""
    size = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            line = line.strip()  # one label per line
            labels.update([line])
            size = i
    return size + 1

# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()
    print("程序开始运行 !")

    # ---------- step0: 命令行交互 ---------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
    parser.add_argument('--min_count_tag', default=1, help="Minimum count for labels in the dataset", type=int)
    parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")
    args = parser.parse_args()
    pprint(args)

    # ---------- step1: 构建词表 ---------- #
    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    # Build label vocab with train and test datasets
    print("Building label vocabulary...")
    labels = Counter()
    size_train_tags = update_labels(os.path.join(args.data_dir, 'train/labels.txt'), labels)
    size_test_tags = update_labels(os.path.join(args.data_dir, 'test/labels.txt'), labels)
    print("- done.")

    # Assert same number of examples in datasets
    assert size_train_sentences == size_train_tags
    assert size_test_sentences == size_test_tags

    # Only keep most frequent tokens
    words = sorted([tok for tok, count in words.items() if count >= args.min_count_word])
    labels = sorted([tok for tok, count in labels.items() if count >= args.min_count_tag])

    # Save vocabularies to text file
    print("Saving vocabularies to file...")
    save_to_txt(words, os.path.join(args.data_dir, 'words.txt'))
    save_to_txt(labels, os.path.join(args.data_dir, 'labels.txt'))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'num_tags': len(labels)
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("-- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    print("程序已结束 !")
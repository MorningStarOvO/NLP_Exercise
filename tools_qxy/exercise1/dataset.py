"""
    本代码用于: 创建数据加载器函数
    创建时间: 2021 年 10 月 10 日
    创建人: MorningStar
    最后一次修改时间: 2021 年 10 月 10 日
"""

# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的 ----- #
import os 

# ----- 模型创建相关的 ----- # 
from torch.utils.data import Dataset

# ----- 图像读取相关 ----- #
from PIL import Image

# ==================== 函数实现 ==================== #
# ---------- 定义数据加载器 ---------- #
class Dataloader(Dataset):
    def __init__(self, path, transform=None, args=None):
        """
            path: 数据集的总路径
            transform: 对图片进行的处理
            args: 命令行交互的内容
        """
        # 赋初值
        self.path = path 
        self.args = args 
        self.transform = transform

        # 找出所有图片的路径
        self.all_paths = []
        self.all_labels = []
        path_list = os.listdir(self.path) 
        for temp_path in path_list:
            # 获得标签值
            temp_str_list = temp_path.split(".")
            temp_label = temp_str_list[0]
            
            # 获得图片路径
            temp = os.path.join(self.path, temp_path)

            # 加入数组
            self.all_paths.append(temp)
            self.all_labels.append(temp_label)

    def __getitem__(self, index):
        # 读取图片
        img_path = self.all_paths[index]  
        image = Image.open(img_path).convert('RGB')
        # if self.args.model == "RNN":
        #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     # 转换为灰度图
        #     image = Image.open(img_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        # 获得标签值
        # cat: 0, dog: 1
        img_label_str = self.all_labels[index]
        if img_label_str == "cat":
            img_label = 0
        elif img_label_str == "dog":
            img_label = 1

        return image, img_label


    def __len__(self):
        return len(self.all_labels)

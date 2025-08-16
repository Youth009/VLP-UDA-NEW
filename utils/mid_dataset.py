# utils/mid_dataset.py
import os
from torch.utils.data import Dataset

class MidDatasetWithWeight(Dataset):
    def __init__(self, base_dataset, weight_dict):
        """
        base_dataset: 任何返回 (img, label) 的 Dataset
        weight_dict: 以 basename(path) 为 key, 判别器分数为 value 的 dict
        """
        self.base = base_dataset
        # 假设 base_dataset.samples[idx][0] 是文件路径
        self.samples = base_dataset.samples
        self.weight_dict = weight_dict

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        path = self.samples[idx][0]
        # 如果找不到，就用权重 1.0
        w = self.weight_dict.get(os.path.basename(path), 1.0)
        return img, label, w

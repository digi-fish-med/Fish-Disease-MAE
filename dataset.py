# coding=utf-8
import os
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class InMemoryImageFolder(Dataset):
    """
    将整个数据集加载到内存中以加速训练。
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        if not os.path.exists(root):
            raise FileNotFoundError(f"Path not found: {root}")

        # 使用 ImageFolder 获取路径和标签
        temp_dataset = ImageFolder(root)
        print(f"Pre-loading {len(temp_dataset)} images into RAM...")

        for path, label in tqdm(temp_dataset.samples, desc="Loading Data"):
            try:
                # 复制图像对象，避免文件句柄未关闭问题
                image = default_loader(path)
                self.samples.append((image.copy(), label))
                image.close()
            except Exception as e:
                print(f"Error loading {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, label = self.samples[index]
        if self.transform:
            image = self.transform(image)
        return image, label
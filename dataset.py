from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(CURRENT_DIR, "data")

def data_preprocess(val_ratio=0.1, use_half_data=True):
    # 训练集：带数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # 验证/测试集：无增强
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # 读取完整数据集
    full_train_val = datasets.CIFAR10(
        root=DATA_FOLDER, train=True, download=False, transform=train_transform
    )

    full_test = datasets.CIFAR10(
        root=DATA_FOLDER, train=False, download=False, transform=test_transform
    )

    # 关键：只取一半训练数据
    if use_half_data:
        half_len = len(full_train_val) // 2
        full_train_val, _ = random_split(full_train_val, [half_len, len(full_train_val) - half_len])

    # 再划分训练集和验证集
    val_size = int(len(full_train_val) * val_ratio)
    train_size = len(full_train_val) - val_size
    train, val = random_split(full_train_val, [train_size, val_size])

    return train, val, full_test

def create_dataloaders(train, val, test):
    train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader
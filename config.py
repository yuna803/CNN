import torch


# 强制使用CPU以避免内存问题
DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
NUM_CLASSES = 10

CLASS_NAMES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]
#

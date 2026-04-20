import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy
from config import DEVICE, LR
from model import CNN

# 全局设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 【优化1：优化器全部重新调整，解决过拟合】 =====================
def get_optimizer(model, name):
    if name == 'sgd':
        # 保持不变，它现在表现很好
        return optim.SGD(model.parameters(), lr=0.001)

    elif name == 'momentum':
        # 降低学习率 + 提高权重衰减，防止后期过拟合
        return optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)

    elif name == 'rmsprop':
        # 大幅降低学习率 + 强权重衰减，解决泛化鸿沟
        return optim.RMSprop(model.parameters(), lr=0.0003, weight_decay=1e-4)

    elif name == 'adam':
        # 大幅降低学习率 + 强权重衰减，解决过拟合
        return optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)


# ===================== 【优化2：早停耐心从3→5，更稳定】 =====================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_epoch(model, loader, opt, cri):
    model.train()
    loss, cor, tot = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(x)
        l = cri(pred, y)
        l.backward()
        opt.step()

        torch.cuda.empty_cache()

        loss += l.item()
        cor += (pred.argmax(1) == y).sum().item()
        tot += y.size(0)
    return loss / len(loader), cor / tot


def val_epoch(model, loader, cri):
    model.eval()
    loss, cor, tot = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss += cri(pred, y).item()
            cor += (pred.argmax(1) == y).sum().item()
            tot += y.size(0)
    return loss / len(loader), cor / tot


def train_one_opt(model, train_loader, val_loader, opt_name):
    cri = nn.CrossEntropyLoss()
    opt = get_optimizer(model, opt_name)
    es = EarlyStopping()
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(15):
        tl, ta = train_epoch(model, train_loader, opt, cri)
        vl, va = val_epoch(model, val_loader, cri)

        hist['train_loss'].append(tl)
        hist['train_acc'].append(ta)
        hist['val_loss'].append(vl)
        hist['val_acc'].append(va)

        print(f"Ep{ep + 1:2d} | Train {tl:.4f} {ta:.4f} | Val {vl:.4f} {va:.4f}")

        if es(vl, model):
            print("触发早停")
            break
    model.load_state_dict(es.best_state)
    return hist, model


def train_all_opt(train_loader, val_loader):
    res = {}
    for name in ['sgd', 'momentum', 'rmsprop', 'adam']:
        print("\n=== 训练", name, "===")
        model = CNN().to(DEVICE)
        h, m = train_one_opt(model, train_loader, val_loader, name)
        res[name] = {'history': h, 'model': m}
        torch.cuda.empty_cache()
    return res


# ===================== 【优化3：标准论文级图表，规范、清晰、无乱码】 =====================
def plot_curves(res):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    colors = {
        'sgd': '#1f77b4',
        'momentum': '#ff7f0e',
        'rmsprop': '#2ca02c',
        'adam': '#d62728'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 训练损失
    ax = axes[0, 0]
    ax.set_title("训练损失对比", fontsize=14)
    ax.set_xlabel("轮数", fontsize=12)
    ax.set_ylabel("损失", fontsize=12)
    ax.grid(True, alpha=0.3)
    for name, data in res.items():
        epochs = list(range(1, len(data['history']['train_loss']) + 1))
        ax.plot(epochs, data['history']['train_loss'], label=name, color=colors[name], linewidth=2)
    ax.legend()

    # 验证损失
    ax = axes[0, 1]
    ax.set_title("验证损失对比", fontsize=14)
    ax.set_xlabel("轮数", fontsize=12)
    ax.set_ylabel("损失", fontsize=12)
    ax.grid(True, alpha=0.3)
    for name, data in res.items():
        epochs = list(range(1, len(data['history']['val_loss']) + 1))
        ax.plot(epochs, data['history']['val_loss'], label=name, color=colors[name], linestyle='--', linewidth=2)
    ax.legend()

    # 训练准确率
    ax = axes[1, 0]
    ax.set_title("训练准确率对比", fontsize=14)
    ax.set_xlabel("轮数", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.grid(True, alpha=0.3)
    for name, data in res.items():
        epochs = list(range(1, len(data['history']['train_acc']) + 1))
        ax.plot(epochs, data['history']['train_acc'], label=name, color=colors[name], linewidth=2)
    ax.legend()

    # 验证准确率
    ax = axes[1, 1]
    ax.set_title("验证准确率对比", fontsize=14)
    ax.set_xlabel("轮数", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.grid(True, alpha=0.3)
    for name, data in res.items():
        epochs = list(range(1, len(data['history']['val_acc']) + 1))
        ax.plot(epochs, data['history']['val_acc'], label=name, color=colors[name], linestyle='--', linewidth=2)
    ax.legend()

    plt.tight_layout()
    plt.show()


def eval_and_plot(model, test_loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            p = model(x).argmax(1).cpu().numpy()
            yt.extend(y.numpy())
            yp.extend(p)

    acc = np.mean(np.array(yt) == np.array(yp))
    print(f"测试集准确率: {acc:.4f}")

    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()
    return acc
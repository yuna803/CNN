from config import DEVICE
from dataset import data_preprocess, create_dataloaders
from utils import train_all_opt, plot_curves, eval_and_plot
import torch

def main():
    print("使用设备:", DEVICE)
    torch.cuda.empty_cache()

    # train, val, test = data_preprocess()
    train, val, test = data_preprocess(use_half_data=False)#训练全部
    train_loader, val_loader, test_loader = create_dataloaders(train, val, test)
    results = train_all_opt(train_loader, val_loader)
    plot_curves(results)
    best_name = max(results.keys(), key=lambda n: max(results[n]['history']['val_acc']))
    eval_and_plot(results[best_name]['model'], test_loader)

if __name__ == '__main__':
    main()
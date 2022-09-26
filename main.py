import warnings
import torch
from torch import optim
from torch.nn import functional as F

from net import SpatialTransformerNet  # 定义模型结构
from visual import visualize_stn  # 定义可视化代码
from dataset import get_loader  # 定义数据集加载
from loop import Loop  # 定义train和test代码段
from utils import random_seed  # 设定随机种子

random_seed(0)
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpatialTransformerNet().to(device)      # 实例化模型
train_loader, test_loader = get_loader(batch_size=128, num_workers=0)   # 获取数据集
optimizer = optim.SGD(model.parameters(), lr=0.01)      # 设定优化器

if __name__ == "__main__":
    epoch = 10
    loop = Loop(model=model, train_loader=train_loader, test_loader=test_loader, loss_fn=F.nll_loss, optimizer=optimizer, device=device)
    for epoch in range(1, epoch + 1):
        loop.train(epoch)
        loop.test(epoch)
        visualize_stn(model=model, test_loader=test_loader, idx=epoch)  # 可视化展示STN前后的图，结果保存在visual/文件夹下
    loop.show() # 绘制Test Acc变化曲线，保存到result.jpg

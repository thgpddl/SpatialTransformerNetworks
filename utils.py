import random
import numpy as np
import torch

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 使用确定性的操作
    torch.backends.cudnn.benchmark = False  # 关闭卷积优化
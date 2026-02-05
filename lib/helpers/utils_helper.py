import torch
import numpy as np
import random



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_cudnn(benchmark=True, deterministic=False):
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
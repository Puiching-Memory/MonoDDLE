import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=4):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # prepare dataloader
    num_workers = cfg.get('num_workers', workers)
    pin_memory = cfg.get('pin_memory', True)
    persistent_workers = cfg.get('persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = cfg.get('prefetch_factor', 2) if num_workers > 0 else None

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_set, shuffle=False, drop_last=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=num_workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=shuffle,
                              sampler=train_sampler,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              prefetch_factor=prefetch_factor,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=num_workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             sampler=test_sampler,
                             pin_memory=pin_memory,
                             persistent_workers=persistent_workers,
                             prefetch_factor=prefetch_factor,
                             drop_last=False)

    return train_loader, test_loader

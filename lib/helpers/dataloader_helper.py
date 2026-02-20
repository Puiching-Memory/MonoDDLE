import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=None, distributed=False):
    """Build train and test dataloaders.

    Parameters
    ----------
    cfg : dict
        Dataset configuration from YAML.  If ``cfg['num_workers']`` is set it
        takes precedence over the *workers* argument.
    workers : int or None
        Number of dataloader workers **per process**.  Defaults to
        ``cfg['num_workers']`` if present, otherwise 4.
    distributed : bool
        If True, use ``DistributedSampler`` for the training set so that each
        DDP rank sees a disjoint partition of the data.  In this mode
        ``cfg['batch_size']`` is treated as the **per-GPU** batch size.

    Returns
    -------
    train_loader, test_loader, train_sampler
        ``train_sampler`` is ``None`` when ``distributed=False``.  In DDP mode
        the caller must call ``train_sampler.set_epoch(epoch)`` every epoch.
    """
    # resolve num_workers: YAML value > explicit arg > default 4
    workers = cfg.get('num_workers', workers if workers is not None else 4)

    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split='train', cfg=cfg)
        test_set = KITTI_Dataset(split='val', cfg=cfg)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              pin_memory=False,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader, train_sampler

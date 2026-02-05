import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import fire
import datetime
import torch
import torch.distributed as dist

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.logger_helper import MonoDDLELogger, console
from lib.helpers.utils_helper import set_random_seed, set_cudnn


def _init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def main(config, e=False, amp=None, compile=None):
    assert (os.path.exists(config))
    cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    distributed, rank, world_size, local_rank = _init_distributed()
    set_random_seed(cfg.get('random_seed', 444))
    set_cudnn()
    run_dir = os.path.join('runs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, 'train_%s.log' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = MonoDDLELogger(log_file if rank == 0 else None, rank=rank)
    cfg.setdefault('trainer', {})['output_dir'] = run_dir
    
    if amp is not None:
        cfg['trainer']['amp'] = amp
    else:
        cfg['trainer'].setdefault('amp', False)
        
    if compile is not None:
        cfg['trainer']['compile'] = compile
    else:
        cfg['trainer'].setdefault('compile', False)

    cfg.setdefault('tester', {})['output_dir'] = run_dir


    # build dataloader
    train_loader, test_loader  = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'])

    if rank == 0:
        try:
            from torch.utils.flop_counter import FlopCounterMode
            has_flop_counter = True
        except ImportError:
            has_flop_counter = False
            logger.warning("torch.utils.flop_counter not found. Skipping FLOPs calculation.")
            
        if has_flop_counter:
            try:
                logger.info("Calculating model FLOPs with torch.utils.flop_counter ...")
                
                # Prepare dummy input
                resolution = cfg['dataset'].get('resolution', [384, 1280]) # H, W
                # CenterNet3D input: (B, C, H, W)
                dummy_input = torch.randn(1, 3, resolution[0], resolution[1])
                
                # Use GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                dummy_input = dummy_input.to(device)
                
                # Context manager for counting
                with FlopCounterMode(display=True):
                    model(dummy_input)
                
                # Move model back to CPU to maintain consistency with subsequent code
                model.to('cpu')
                del dummy_input
                torch.cuda.empty_cache()
            except Exception as flop_error:
                logger.warning(f"Error calculating FLOPs: {flop_error}")
                model.to('cpu') # Ensure model is back on CPU

    if e:
        logger.print_section('Evaluation Only')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        if rank == 0:
            tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)


    logger.print_section('Training')
    logger.print_config({
        'Batch Size': cfg['dataset']['batch_size'],
        'Learning Rate': cfg['optimizer']['lr'],
        'Max Epochs': cfg['trainer'].get('max_epoch', 'N/A'),
        'Output Dir': run_dir,
    }, title='Training Configuration')
    
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    trainer.train()

    if rank == 0:
        logger.print_section('Final Evaluation')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()


if __name__ == '__main__':
    fire.Fire(main)
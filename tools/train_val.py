import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model, log_model_complexity
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='End-to-End Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')

args = parser.parse_args()


def _build_experiment_paths(config_path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    configs_root = os.path.join(ROOT_DIR, 'experiments', 'configs')
    config_abs = os.path.abspath(config_path)

    if os.path.commonpath([config_abs, configs_root]) == configs_root:
        rel_no_ext = os.path.splitext(os.path.relpath(config_abs, configs_root))[0]
    else:
        rel_no_ext = os.path.splitext(os.path.basename(config_abs))[0]

    run_dir = os.path.join(ROOT_DIR, 'experiments', 'results', rel_no_ext, ts)
    return {
        'run_dir': run_dir,
        'log_dir': os.path.join(run_dir, 'logs'),
        'ckpt_dir': os.path.join(run_dir, 'checkpoints'),
        'output_dir': os.path.join(run_dir, 'outputs'),
    }



def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    exp_paths = _build_experiment_paths(args.config)
    os.makedirs(exp_paths['log_dir'], exist_ok=True)
    os.makedirs(exp_paths['ckpt_dir'], exist_ok=True)
    os.makedirs(exp_paths['output_dir'], exist_ok=True)

    cfg.setdefault('trainer', {})
    cfg.setdefault('tester', {})
    cfg['trainer']['checkpoints_dir'] = exp_paths['ckpt_dir']
    cfg['tester']['output_dir'] = exp_paths['output_dir']

    if not args.evaluate_only:
        cfg['tester']['checkpoint'] = os.path.join(
            exp_paths['ckpt_dir'],
            'checkpoint_epoch_%d.pth' % cfg['trainer'].get('max_epoch', 140)
        )
        cfg['tester']['checkpoints_dir'] = exp_paths['ckpt_dir']

    # resolve root_dir relative to the project root when given as a relative path
    root_dir = cfg['dataset'].get('root_dir', 'data/KITTI')
    if not os.path.isabs(root_dir):
        cfg['dataset']['root_dir'] = os.path.normpath(os.path.join(ROOT_DIR, root_dir))

    set_random_seed(cfg.get('random_seed', 444))
    log_file = os.path.join(exp_paths['log_dir'], 'train.log')
    logger = create_logger(log_file)
    logger.info('Experiment dir: %s' % exp_paths['run_dir'])


    # build dataloader
    train_loader, test_loader, _ = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'])

    profile_resolution = tuple(train_loader.dataset.resolution[::-1].tolist())
    log_model_complexity(model, logger=logger, input_resolution=profile_resolution)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)


    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      distill_cfg=cfg.get('distill', None))
    trainer.train()

    logger.info('###################  Evaluation  ##################' )
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger)
    tester.test()


if __name__ == '__main__':
    main()
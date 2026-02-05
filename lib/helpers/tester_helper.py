import os

import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.logger_helper import create_progress_bar, console



class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = cfg.get('output_dir', './outputs')
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.eval = eval


    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            if self.cfg.get('checkpoint'):
                assert os.path.exists(self.cfg['checkpoint'])
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=self.cfg['checkpoint'],
                                map_location=self.device,
                                logger=self.logger)
            self.model.to(self.device)
            self.inference()
            return self.evaluate()

        # test all checkpoints in the given dir
        if self.cfg['mode'] == 'all':
            checkpoints_list = []
            for _, _, files in os.walk(self.cfg['checkpoints_dir']):
                checkpoints_list = [os.path.join(self.cfg['checkpoints_dir'], f) for f in files if f.endswith(".pth")]
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()
            return None



    def inference(self):
        self.model.eval()

        results = {}
        with create_progress_bar(description="Evaluation") as progress:
            eval_task = progress.add_task("[cyan]Evaluating", total=len(self.dataloader))
            
            with torch.inference_mode():
                for batch_idx, (inputs, _, info) in enumerate(self.dataloader):
                    # load evaluation data and move data to GPU.
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)

                    # get corresponding calibs & transform tensor to numpy
                    calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
                    cls_mean_size = self.dataloader.dataset.cls_mean_size
                    dets = decode_detections(dets=dets,
                                             info=info,
                                             calibs=calibs,
                                             cls_mean_size=cls_mean_size,
                                             threshold=self.cfg.get('threshold', 0.2))
                    results.update(dets)
                    progress.update(eval_task, advance=1)

        # save the result for evaluation.
        self.logger.info('Saving evaluation results...')
        self.save_results(results, output_dir=self.output_dir)



    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in self.dataloader.dataset.idx_list:
            img_id = int(img_id)
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            if img_id not in results or results[img_id] is None:
                f.close()
                continue
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()



    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'data')
        return self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)


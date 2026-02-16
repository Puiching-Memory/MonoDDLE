import os
import tqdm

import torch

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections



class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, eval=False, epoch=None):
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
        self.epoch = epoch
        self._vis_batches = []  # store first few batches for visualization
        self._max_vis_batches = cfg.get('max_vis_batches', 2)


    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single':
            if self.cfg.get('checkpoint'):
                checkpoint_path = self.cfg['checkpoint']
                assert os.path.exists(checkpoint_path), \
                    "Checkpoint not found: %s" % checkpoint_path
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint_path,
                                map_location=self.device,
                                logger=self.logger)
            self.model.to(self.device)
            self.inference()
            return self.evaluate()

        # test all checkpoints in the given dir
        if self.cfg['mode'] == 'all':
            checkpoints_list = []
            checkpoints_dir = self.cfg.get('checkpoints_dir', os.path.join(self.output_dir, 'weights'))
            for _, _, files in os.walk(checkpoints_dir):
                checkpoints_list = [os.path.join(checkpoints_dir, f)
                                  for f in files if f.endswith(".pth")]
            checkpoints_list.sort(key=lambda x: os.path.getmtime(x))

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
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        self._vis_batches = []
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        for batch_idx, (inputs, _, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(dets=dets,
                                     info=info,
                                     calibs=calibs,
                                     cls_mean_size=cls_mean_size,
                                     threshold=self.cfg.get('threshold', 0.2))
            results.update(dets)

            # Collect first few batches for visualization
            if len(self._vis_batches) < self._max_vis_batches:
                self._vis_batches.append({
                    'inputs': inputs.detach().cpu(),
                    'outputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in outputs.items()},
                    'info': info,
                    'calibs': calibs,
                })

            progress_bar.update()

        progress_bar.close()
        torch.set_grad_enabled(True)

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

        # generate visualizations
        self._visualize()



    def save_results(self, results, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        output_dir = os.path.join(output_dir, 'kitti_eval', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in self.dataloader.dataset.idx_list:
            img_id_int = int(img_id)
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id_int))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id_int)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id_int),
                                           self.dataloader.dataset.get_sample_token(img_id_int) + '.txt')

            f = open(output_path, 'w')
            if img_id_int not in results or results[img_id_int] is None:
                f.close()
                continue
            for i in range(len(results[img_id_int])):
                class_name = self.class_name[int(results[img_id_int][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id_int][i])):
                    f.write(' {:.2f}'.format(results[img_id_int][i][j]))
                f.write('\n')
            f.close()



    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'kitti_eval', 'data')
        return self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger, epoch=self.epoch, eval_only=self.eval)

    def _visualize(self):
        """Generate 2D/3D box and heatmap visualizations for collected batches."""
        if not self._vis_batches:
            return

        try:
            from lib.helpers.visualization_helper import visualize_results
        except ImportError:
            self.logger.warning('visualization_helper not found, skipping visualization.')
            return

        epoch_tag = self.epoch if self.epoch is not None else 0
        vis_dir = os.path.join(self.output_dir, 'visualizations', 'epoch_%d' % epoch_tag)
        os.makedirs(vis_dir, exist_ok=True)

        dataset = self.dataloader.dataset
        cls_mean_size = dataset.cls_mean_size

        try:
            for batch_data in self._vis_batches:
                inputs = batch_data['inputs']
                outputs = batch_data['outputs']
                info = batch_data['info']
                calibs = batch_data['calibs']

                # Collect ground-truth objects for each image in the batch
                gt_objects = []
                for img_id in info['img_id']:
                    try:
                        gt_objects.append(dataset.get_label(int(img_id)))
                    except Exception:
                        gt_objects.append([])

                visualize_results(
                    images=inputs,
                    outputs=outputs,
                    targets=None,
                    info=info,
                    calibs=calibs,
                    cls_mean_size=cls_mean_size,
                    threshold=self.cfg.get('threshold', 0.2),
                    output_dir=vis_dir,
                    epoch=epoch_tag,
                    gt_objects=gt_objects,
                    dataset=dataset,
                    writelist=getattr(dataset, 'writelist', None),
                )

            self.logger.info('Visualizations saved to %s' % vis_dir)
        except Exception as e:
            self.logger.warning('Visualization failed: %s' % str(e))

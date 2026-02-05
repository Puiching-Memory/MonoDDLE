import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle


def class2angle_torch(cls, residual, to_label_format=False):
    angle_per_class = 2 * np.pi / float(12)
    angle_center = cls.float() * angle_per_class
    angle = angle_center + residual
    if to_label_format:
        angle = torch.where(angle > np.pi, angle - 2 * np.pi, angle)
    return angle


def get_heading_angle_torch(heading):
    heading_bin = heading[:, 0:12]
    heading_res = heading[:, 12:24]
    cls = torch.argmax(heading_bin, dim=1)
    res = heading_res.gather(1, cls.view(-1, 1)).squeeze(1)
    return class2angle_torch(cls, res, to_label_format=True)


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    input: dets, torch tensor [batch x max_dets x dim]
    input: img_info, dict
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    if not torch.is_tensor(dets):
        dets = torch.as_tensor(dets, device='cuda', dtype=torch.float32)
    device = dets.device

    bbox_downsample_ratio = info['bbox_downsample_ratio']
    if not torch.is_tensor(bbox_downsample_ratio):
        bbox_downsample_ratio = torch.as_tensor(bbox_downsample_ratio, device=device, dtype=torch.float32)
    img_ids = info['img_id']
    if not torch.is_tensor(img_ids):
        img_ids = torch.as_tensor(img_ids, device=device)

    cls_mean_size = torch.as_tensor(cls_mean_size, device=device, dtype=torch.float32)

    scores = dets[:, :, 1]
    valid_mask = scores >= threshold

    results = {}
    for i in range(dets.shape[0]):
        preds = []
        valid_inds = torch.nonzero(valid_mask[i], as_tuple=False).squeeze(1)
        if valid_inds.numel() == 0:
            results[int(img_ids[i].item())] = preds
            continue

        di = dets[i, valid_inds]
        cls_id = di[:, 0].long()
        score = di[:, 1] * di[:, -1]

        ratio = bbox_downsample_ratio[i]
        x = di[:, 2] * ratio[0]
        y = di[:, 3] * ratio[1]
        w = di[:, 4] * ratio[0]
        h = di[:, 5] * ratio[1]
        bbox = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=1)

        depth = di[:, 6]
        dimensions = di[:, 31:34] + cls_mean_size[cls_id]
        x3d = di[:, 34] * ratio[0]
        y3d = di[:, 35] * ratio[1]
        heading = di[:, 7:31]
        alpha = get_heading_angle_torch(heading)

        cls_id_cpu = cls_id.detach().cpu().numpy()
        score_cpu = score.detach().cpu().numpy()
        bbox_cpu = bbox.detach().cpu().numpy()
        dimensions_cpu = dimensions.detach().cpu().numpy()
        x3d_cpu = x3d.detach().cpu().numpy()
        y3d_cpu = y3d.detach().cpu().numpy()
        depth_cpu = depth.detach().cpu().numpy()
        alpha_cpu = alpha.detach().cpu().numpy()

        for j in range(len(cls_id_cpu)):
            cls_val = int(cls_id_cpu[j])
            x3d_val = float(x3d_cpu[j])
            y3d_val = float(y3d_cpu[j])
            depth_val = float(depth_cpu[j])

            locations = calibs[i].img_to_rect(np.array([x3d_val]), np.array([y3d_val]), np.array([depth_val])).reshape(-1)
            locations[1] += dimensions_cpu[j][0] / 2

            alpha_val = float(alpha_cpu[j])
            ry = calibs[i].alpha2ry(alpha_val, x3d_val)

            preds.append([
                cls_val,
                alpha_val,
                bbox_cpu[j][0],
                bbox_cpu[j][1],
                bbox_cpu[j][2],
                bbox_cpu[j][3],
                dimensions_cpu[j][0],
                dimensions_cpu[j][1],
                dimensions_cpu[j][2],
                locations[0],
                locations[1],
                locations[2],
                ry,
                float(score_cpu[j]),
            ])

        results[int(img_ids[i].item())] = preds
    return results


def extract_dets_from_outputs(outputs, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    heading = outputs['heading']
    depth = outputs['depth'][:, 0:1, :, :]
    sigma = outputs['depth'][:, 1:2, :, :]
    size_3d = outputs['size_3d']
    offset_3d = outputs['offset_3d']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    depth = 1. / (depth.sigmoid() + 1e-6) - 1.
    sigma = torch.exp(-sigma)

    batch, channel, height, width = heatmap.size() # get shape

    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    offset_3d = _transpose_and_gather_feat(offset_3d, inds)
    offset_3d = offset_3d.view(batch, K, 2)
    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    heading = _transpose_and_gather_feat(heading, inds)
    heading = heading.view(batch, K, 24)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    sigma = _transpose_and_gather_feat(sigma, inds)
    sigma = sigma.view(batch, K, 1)
    size_3d = _transpose_and_gather_feat(size_3d, inds)
    size_3d = size_3d.view(batch, K, 3)
    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=2)

    return detections



############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from torch.utils.data import DataLoader

    dataset = KITTI_Dataset('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)

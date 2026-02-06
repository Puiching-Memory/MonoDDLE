import numpy as np
import torch
import math

import os
from torch.utils.cpp_extension import load

try:
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sources = [os.path.join(src_dir, 'iou3d.cpp'), os.path.join(src_dir, 'iou3d_kernel.cu')]
    
    iou3d_cuda = load(
        name='iou3d_cuda',
        sources=sources,
        verbose=True
    )
except Exception as e:
    print(f"Error JIT compiling iou3d_cuda: {e}")
    raise

DISTANCE_COVER = False

def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

def clean_data(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower() and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False
        if ((gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty])
                or (gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty])
                or (height <= MIN_HEIGHT[difficulty])):
            ignore = True
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])
    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

def clean_data_by_distance(gt_anno, dt_anno, current_class, difficulty):
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    MAX_OCCLUSION = [0, 1, 2]
    MIN_HEIGHT = [40, 25, 25]
    MAX_DISTANCE = [30, 50, 70]
    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1
        if (gt_name == current_cls_name):
            valid_class = 1
        elif (current_cls_name == "Pedestrian".lower()
              and "Person_sitting".lower() == gt_name):
            valid_class = 0
        elif (current_cls_name == "Car".lower() and "Van".lower() == gt_name):
            valid_class = 0
        else:
            valid_class = -1
        ignore = False

        dis = np.linalg.norm(gt_anno["location"][i])
        if DISTANCE_COVER:
            if ((gt_anno["occluded"][i] > MAX_OCCLUSION[2]) or
                    (gt_anno["truncated"][i] > MAX_TRUNCATION[2]) or
                    (height <= MIN_HEIGHT[2]) or
                    (dis > MAX_DISTANCE[difficulty])):
                ignore = True
        else:
            if difficulty == 0:
                if ((gt_anno["occluded"][i] > MAX_OCCLUSION[2]) or
                        (gt_anno["truncated"][i] > MAX_TRUNCATION[2]) or
                        (height <= MIN_HEIGHT[2]) or
                        (dis > MAX_DISTANCE[difficulty])):
                    ignore = True
            else:
                if ((gt_anno["occluded"][i] > MAX_OCCLUSION[2]) or
                        (gt_anno["truncated"][i] > MAX_TRUNCATION[2]) or
                        (height <= MIN_HEIGHT[2]) or
                        (dis > MAX_DISTANCE[difficulty]) or
                        (dis <= MAX_DISTANCE[difficulty - 1])):
                    ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif (valid_class == 0 or (ignore and (valid_class == 1))):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        if (dt_anno["name"][i].lower() == current_cls_name):
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        if height < MIN_HEIGHT[2]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes

def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    boxes_t = torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous()
    query_t = torch.as_tensor(query_boxes, device="cuda", dtype=torch.float32).contiguous()
    out = iou3d_cuda.iou2d_gpu(boxes_t, query_t, criterion)
    return out.cpu().numpy().astype(np.float64 if boxes.dtype == np.float64 else np.float32)

def bev_box_overlap(boxes, qboxes, criterion=-1):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    boxes_t = torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous()
    qboxes_t = torch.as_tensor(qboxes, device="cuda", dtype=torch.float32).contiguous()
    out = iou3d_cuda.rotate_iou_gpu(boxes_t, qboxes_t, criterion)
    return out.cpu().numpy().astype(np.float64 if boxes.dtype == np.float64 else np.float32)

def d3_box_overlap(boxes, qboxes, criterion=-1):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=boxes.dtype)
    boxes_t = torch.as_tensor(boxes, device="cuda", dtype=torch.float32).contiguous()
    qboxes_t = torch.as_tensor(qboxes, device="cuda", dtype=torch.float32).contiguous()
    out = iou3d_cuda.d3_box_overlap_gpu(boxes_t, qboxes_t, criterion)
    return out.cpu().numpy().astype(np.float64 if boxes.dtype == np.float64 else np.float32)

def compute_statistics_jit(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det, dc_bboxes, metric, min_overlap, thresh=0, compute_aos=False, candidates=None):
    overlaps_t = torch.from_numpy(overlaps).float()
    gt_datas_t = torch.from_numpy(gt_datas).float()
    dt_datas_t = torch.from_numpy(dt_datas).float()
    ignored_gt_t = torch.from_numpy(ignored_gt).long()
    ignored_det_t = torch.from_numpy(ignored_det).long()
    dc_bboxes_t = torch.from_numpy(dc_bboxes).float()
    
    tp_t, fp_t, fn_t, sim_t, thresh_t = iou3d_cuda.compute_statistics_cpp(
        overlaps_t, 
        gt_datas_t, 
        dt_datas_t, 
        ignored_gt_t, 
        ignored_det_t, 
        dc_bboxes_t, 
        metric, 
        float(min_overlap), 
        float(thresh), 
        compute_aos
    )
    return tp_t.item(), fp_t.item(), fn_t.item(), sim_t.item(), thresh_t.numpy()

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if same_part == 0:
        return [num]
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas, dt_datas, dontcares, ignored_gts, ignored_dets, metric, min_overlap, thresholds, compute_aos=False):
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]
        gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
        dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
        ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
        ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
        dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
        
        for t, thresh in enumerate(thresholds):
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap, gt_data, dt_data, ignored_gt, ignored_det, dontcare,
                metric, min_overlap=min_overlap, thresh=thresh, compute_aos=compute_aos
            )
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]

def calculate_iou_partly(gt_annos, dt_annos, metric, num_parts=50):
    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_box_num = total_gt_num[example_idx:example_idx+num_part]
        dt_box_num = total_dt_num[example_idx:example_idx+num_part]
        
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_n = gt_box_num[i]
            dt_n = dt_box_num[i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_n, dt_num_idx:dt_num_idx + dt_n]
            )
            gt_num_idx += gt_n
            dt_num_idx += dt_n
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num

def _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=True):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty) if DIForDIS \
            else clean_data_by_distance(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = rets
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate(
            [gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis],
            dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
            total_dc_num, total_num_valid_gt)

def eval_class(gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps, compute_aos=False, num_parts=50, DIForDIS=True):
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)
    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    
    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty, DIForDIS=DIForDIS)
            (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets,
             dontcares, total_dc_num, total_num_valid_gt) = rets
            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                thresholdss = []
                for i in range(len(gt_annos)):
                    tp, fp, fn, similarity, thresholds = compute_statistics_jit(
                        overlaps[i], gt_datas_list[i], dt_datas_list[i],
                        ignored_gts[i], ignored_dets[i], dontcares[i],
                        metric, min_overlap=min_overlap, thresh=0.0
                    )
                    thresholdss += thresholds.tolist()
                
                thresholdss = np.array(thresholdss)
                thresholds = get_thresholds(thresholdss, total_num_valid_gt)
                thresholds = np.array(thresholds)
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, num_part in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx:idx + num_part], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx:idx + num_part], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + num_part], 0)
                    ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_part], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_part], 0)
                    
                    fused_compute_statistics(
                        parted_overlaps[j], pr,
                        total_gt_num[idx:idx + num_part], total_dt_num[idx:idx + num_part], total_dc_num[idx:idx + num_part],
                        gt_datas_part, dt_datas_part, dc_datas_part,
                        ignored_gts_part, ignored_dets_part,
                        metric, min_overlap=min_overlap, thresholds=thresholds, compute_aos=compute_aos
                    )
                    idx += num_part
                
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict

def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100

def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100

def do_eval(gt_annos, dt_annos, current_classes, min_overlaps, compute_aos=False, PR_detail_dict=None, DIForDIS=True):
    difficultys = [0, 1, 2]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps, compute_aos, DIForDIS=DIForDIS)
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['bbox'] = ret['precision']

    mAP_aos = mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])
        if PR_detail_dict is not None:
            PR_detail_dict['aos'] = ret['orientation']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps, DIForDIS=DIForDIS)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['bev'] = ret['precision']

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps, DIForDIS=DIForDIS)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])
    if PR_detail_dict is not None:
        PR_detail_dict['3d'] = ret['precision']
        
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40

def do_coco_style_eval(gt_annos, dt_annos, current_classes, overlap_ranges, compute_aos):
    min_overlaps = np.zeros([10, *overlap_ranges.shape[1:]])
    for i in range(overlap_ranges.shape[1]):
        for j in range(overlap_ranges.shape[2]):
            min_overlaps[:, i, j] = np.linspace(*overlap_ranges[:, i, j])
    mAP_bbox, mAP_bev, mAP_3d, mAP_aos, _, _, _, _ = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos)
    mAP_bbox = mAP_bbox.mean(-1)
    mAP_bev = mAP_bev.mean(-1)
    mAP_3d = mAP_3d.mean(-1)
    if mAP_aos is not None:
        mAP_aos = mAP_aos.mean(-1)
    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos

def get_official_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7], 
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'}
    name_to_class = {v: n for n, v in class_to_name.items()}
    
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict, DIForDIS=True)

    ret_dict = {}
    rich_data = []
    result = '' 

    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            overlap_str = "{:.2f}, {:.2f}, {:.2f}".format(*min_overlaps[i, :, j])
            
            block_data = {
                'class_name': class_to_name[curcls],
                'overlap_str': overlap_str,
                'metrics': {},
                'metrics_R40': {}
            }
            
            block_data['metrics']['bbox'] = mAPbbox[j, :, i]
            block_data['metrics']['bev'] = mAPbev[j, :, i]
            block_data['metrics']['3d'] = mAP3d[j, :, i]
            
            if compute_aos:
                block_data['metrics']['aos'] = mAPaos[j, :, i]
                if i == 0:
                    ret_dict[f'{class_to_name[curcls]}_aos_easy'] = mAPaos[j, 0, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_moderate'] = mAPaos[j, 1, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_hard'] = mAPaos[j, 2, 0]

            block_data['metrics_R40']['bbox'] = mAPbbox_R40[j, :, i]
            block_data['metrics_R40']['bev'] = mAPbev_R40[j, :, i]
            block_data['metrics_R40']['3d'] = mAP3d_R40[j, :, i]

            if compute_aos:
                block_data['metrics_R40']['aos'] = mAPaos_R40[j, :, i]
                if i == 0:
                    ret_dict[f'{class_to_name[curcls]}_aos_easy_R40'] = mAPaos_R40[j, 0, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_moderate_R40'] = mAPaos_R40[j, 1, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_hard_R40'] = mAPaos_R40[j, 2, 0]
            
            rich_data.append(block_data)

            if i == 0:
                cname = class_to_name[curcls]
                ret_dict[f'{cname}_3d_easy'] = mAP3d[j, 0, 0]
                ret_dict[f'{cname}_3d_moderate'] = mAP3d[j, 1, 0]
                ret_dict[f'{cname}_3d_hard'] = mAP3d[j, 2, 0]
                ret_dict[f'{cname}_bev_easy'] = mAPbev[j, 0, 0]
                ret_dict[f'{cname}_bev_moderate'] = mAPbev[j, 1, 0]
                ret_dict[f'{cname}_bev_hard'] = mAPbev[j, 2, 0]
                ret_dict[f'{cname}_image_easy'] = mAPbbox[j, 0, 0]
                ret_dict[f'{cname}_image_moderate'] = mAPbbox[j, 1, 0]
                ret_dict[f'{cname}_image_hard'] = mAPbbox[j, 2, 0]

                ret_dict[f'{cname}_3d_easy_R40'] = mAP3d_R40[j, 0, 0]
                ret_dict[f'{cname}_3d_moderate_R40'] = mAP3d_R40[j, 1, 0]
                ret_dict[f'{cname}_3d_hard_R40'] = mAP3d_R40[j, 2, 0]
                ret_dict[f'{cname}_bev_easy_R40'] = mAPbev_R40[j, 0, 0]
                ret_dict[f'{cname}_bev_moderate_R40'] = mAPbev_R40[j, 1, 0]
                ret_dict[f'{cname}_bev_hard_R40'] = mAPbev_R40[j, 2, 0]
                ret_dict[f'{cname}_image_easy_R40'] = mAPbbox_R40[j, 0, 0]
                ret_dict[f'{cname}_image_moderate_R40'] = mAPbbox_R40[j, 1, 0]
                ret_dict[f'{cname}_image_hard_R40'] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict, rich_data

def get_distance_eval_result(gt_annos, dt_annos, current_classes, PR_detail_dict=None):
    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7], 
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5], 
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    class_to_name = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van', 4: 'Person_sitting', 5: 'Truck'}
    name_to_class = {v: n for n, v in class_to_name.items()}
    
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int
    min_overlaps = min_overlaps[:, :, current_classes]
    result = ''
    
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, min_overlaps, compute_aos, PR_detail_dict=PR_detail_dict, DIForDIS=False)

    ret_dict = {}
    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            result += f"{class_to_name[curcls]} AP@{min_overlaps[i, 0, j]:.2f}, {min_overlaps[i, 1, j]:.2f}, {min_overlaps[i, 2, j]:.2f}:\n"
            result += f"bbox AP:{mAPbbox[j, 0, i]:.4f}, {mAPbbox[j, 1, i]:.4f}, {mAPbbox[j, 2, i]:.4f}\n"
            result += f"bev  AP:{mAPbev[j, 0, i]:.4f}, {mAPbev[j, 1, i]:.4f}, {mAPbev[j, 2, i]:.4f}\n"
            result += f"3d   AP:{mAP3d[j, 0, i]:.4f}, {mAP3d[j, 1, i]:.4f}, {mAP3d[j, 2, i]:.4f}\n"

            if compute_aos:
                result += f"aos  AP:{mAPaos[j, 0, i]:.2f}, {mAPaos[j, 1, i]:.2f}, {mAPaos[j, 2, i]:.2f}\n"
                if i == 0:
                    ret_dict[f'{class_to_name[curcls]}_aos_30m'] = mAPaos[j, 0, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_50m'] = mAPaos[j, 1, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_70m'] = mAPaos[j, 2, 0]

            result += f"{class_to_name[curcls]} AP_R40@{min_overlaps[i, 0, j]:.2f}, {min_overlaps[i, 1, j]:.2f}, {min_overlaps[i, 2, j]:.2f}:\n"
            result += f"bbox AP:{mAPbbox_R40[j, 0, i]:.4f}, {mAPbbox_R40[j, 1, i]:.4f}, {mAPbbox_R40[j, 2, i]:.4f}\n"
            result += f"bev  AP:{mAPbev_R40[j, 0, i]:.4f}, {mAPbev_R40[j, 1, i]:.4f}, {mAPbev_R40[j, 2, i]:.4f}\n"
            result += f"3d   AP:{mAP3d_R40[j, 0, i]:.4f}, {mAP3d_R40[j, 1, i]:.4f}, {mAP3d_R40[j, 2, i]:.4f}\n"
            if compute_aos:
                result += f"aos  AP:{mAPaos_R40[j, 0, i]:.2f}, {mAPaos_R40[j, 1, i]:.2f}, {mAPaos_R40[j, 2, i]:.2f}\n"
                if i == 0:
                    ret_dict[f'{class_to_name[curcls]}_aos_30m_R40'] = mAPaos_R40[j, 0, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_50m_R40'] = mAPaos_R40[j, 1, 0]
                    ret_dict[f'{class_to_name[curcls]}_aos_70m_R40'] = mAPaos_R40[j, 2, 0]

            if i == 0:
                cname = class_to_name[curcls]
                ret_dict[f'{cname}_3d_30m'] = mAP3d[j, 0, 0]
                ret_dict[f'{cname}_3d_50m'] = mAP3d[j, 1, 0]
                ret_dict[f'{cname}_3d_70m'] = mAP3d[j, 2, 0]
                ret_dict[f'{cname}_bev_30m'] = mAPbev[j, 0, 0]
                ret_dict[f'{cname}_bev_50m'] = mAPbev[j, 1, 0]
                ret_dict[f'{cname}_bev_70m'] = mAPbev[j, 2, 0]
                ret_dict[f'{cname}_image_30m'] = mAPbbox[j, 0, 0]
                ret_dict[f'{cname}_image_50m'] = mAPbbox[j, 1, 0]
                ret_dict[f'{cname}_image_70m'] = mAPbbox[j, 2, 0]

                ret_dict[f'{cname}_3d_30m_R40'] = mAP3d_R40[j, 0, 0]
                ret_dict[f'{cname}_3d_50m_R40'] = mAP3d_R40[j, 1, 0]
                ret_dict[f'{cname}_3d_70m_R40'] = mAP3d_R40[j, 2, 0]
                ret_dict[f'{cname}_bev_30m_R40'] = mAPbev_R40[j, 0, 0]
                ret_dict[f'{cname}_bev_50m_R40'] = mAPbev_R40[j, 1, 0]
                ret_dict[f'{cname}_bev_70m_R40'] = mAPbev_R40[j, 2, 0]
                ret_dict[f'{cname}_image_30m_R40'] = mAPbbox_R40[j, 0, 0]
                ret_dict[f'{cname}_image_50m_R40'] = mAPbbox_R40[j, 1, 0]
                ret_dict[f'{cname}_image_70m_R40'] = mAPbbox_R40[j, 2, 0]

    return result, ret_dict

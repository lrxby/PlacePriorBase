# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import get_context

import numpy as np
import torch
from mmcv.ops import box_iou_quadri, box_iou_rotated
from mmdet.evaluation.functional import average_precision
from mmengine.logging import print_log
from terminaltables import AsciiTable


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 box_type='rbox',
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Modified to return matched pairs with IoU for custom metric calculation.
    """
    det_bboxes = np.array(det_bboxes)
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0],
                  dtype=bool), np.ones(gt_bboxes_ignore.shape[0], dtype=bool)))
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    
    # store matched triples: (pred box, GT box, rotated IoU)
    matched_pairs = []

    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
        return tp, fp, matched_pairs

    if box_type == 'rbox':
        ious = box_iou_rotated(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    elif box_type == 'qbox':
        ious = box_iou_quadri(
            torch.from_numpy(det_bboxes).float(),
            torch.from_numpy(gt_bboxes).float()).numpy()
    else:
        raise NotImplementedError
    
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_bboxes[:, -1])
    
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
            
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        # record (pred, GT, IoU) at the primary scale
                        if k == 0:
                            matched_pairs.append((det_bboxes[i], gt_bboxes[matched_gt], ious_max[i]))
                    else:
                        fp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                if box_type == 'rbox':
                    bbox = det_bboxes[i, :5]
                    area = bbox[2] * bbox[3]
                elif box_type == 'qbox':
                    bbox = det_bboxes[i, :8]
                    # keep the original logic (qbox area computation omitted here)
                    area = 1.0 
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
                    
    return tp, fp, matched_pairs

def get_cls_results(det_results, annotations, class_id, box_type):
    """Get det results and gt information of a certain class."""
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        if len(ann.get('bboxes', [])) != 0:
            gt_inds = ann['labels'] == class_id
            cls_gts.append(ann['bboxes'][gt_inds, :])
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            if box_type == 'rbox':
                cls_gts.append(torch.zeros((0, 5), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 5), dtype=torch.float64))
            elif box_type == 'qbox':
                cls_gts.append(torch.zeros((0, 8), dtype=torch.float64))
                cls_gts_ignore.append(torch.zeros((0, 8), dtype=torch.float64))
            else:
                raise NotImplementedError
    return cls_dets, cls_gts, cls_gts_ignore


def eval_rbbox_map(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thr=0.5,
                   use_07_metric=True,
                   box_type='rbox',
                   dataset=None,
                   logger=None,
                   nproc=4):
    """Evaluate mAP of a rotated dataset.
    Modified to return per-class matched pairs with IoU.
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = get_context('spawn').Pool(nproc)
    eval_results = []
    
    # per-class match result lists
    cls_matched_pairs = []

    for i in range(num_classes):
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i, box_type)

        tpfp_res = pool.starmap(
            tpfp_default,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [box_type for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        
        tp, fp, matched_pairs_cls = tuple(zip(*tpfp_res))
        
        # flatten matches of all images into this class
        current_cls_pairs = []
        for img_pairs in matched_pairs_cls:
            current_cls_pairs.extend(img_pairs)
        cls_matched_pairs.append(current_cls_pairs)

        num_gts = np.zeros(num_scales, dtype=int)
        for _, bbox in enumerate(cls_gts):
            num_gts[0] += bbox.shape[0]  # simplified; actually handled by area_ranges

        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
        mode = 'area' if not use_07_metric else '11points'
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': recalls,
            'precision': precisions,
            'ap': ap
        })
    pool.close()
    
    # ... (intermediate mean_ap computation omitted) ...
    aps = [r['ap'] for r in eval_results if r['num_gts'] > 0]
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results, cls_matched_pairs

def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.
    Modified to merge mIoU, mAngle, mSize into the big table.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    # build the table header dynamically
    header = ['class', 'gts', 'dets', 'recall', 'ap']
    extra_keys = ['mIoU', 'mAngle', 'mAngle_longedge', 'mSize', 'mAngle_le_slim']
    present_extras = [k for k in extra_keys if k in results[0]]
    header.extend(present_extras)

    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            
            # fill class rows (N/A shown as-is)
            for k in present_extras:
                v = results[j].get(k, 'N/A')
                if isinstance(v, (int, float)):
                    row_data.append(f"{v:.3f}")
                else:
                    row_data.append(str(v))
                
            table_data.append(row_data)
        
        # fill the footer (mean row)
        footer = ['mAP', '', '', '', f'{mean_ap[i]:.3f}']
        for k in present_extras:
            # only classes with num_gts > 0 and TPs are counted
            valid_values = []
            for j in range(num_classes):
                mIoU_v = results[j].get('mIoU', 0)
                has_tp = isinstance(mIoU_v, (int, float)) and mIoU_v > 0
                v = results[j].get(k)
                if (results[j]['num_gts'] > 0 and has_tp
                        and isinstance(v, (int, float))):
                    valid_values.append(v)
            
            # mean over valid classes
            avg_val = np.mean(valid_values) if valid_values else 'N/A'
            if isinstance(avg_val, float):
                footer.append(f'{avg_val:.3f}')
            else:
                footer.append('N/A')
            
        table_data.append(footer)
        
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
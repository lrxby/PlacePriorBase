# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import tempfile
import re
import zipfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmdet.structures.bbox import bbox_overlaps
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from mmengine.logging import print_log 

# print function imported from mean_ap
from mmrotate.evaluation.functional.mean_ap import print_map_summary 
from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox


@METRICS.register_module()
class DOTAMetric(BaseMetric):
    """DOTA evaluation metric.
    
    Modified to support mIoU, mAngle, and mSize metrics integrated into the big table.
    """
    
    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 square_cls: Optional[list] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not None'
        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr
        self.use_07_metric = True if eval_mode == '11points' else False
        # direction-agnostic classes (circular/near-square): excluded from global stats; default DOTA square classes
        self.square_cls = square_cls if square_cls is not None else [1, 9, 11]

    # ---------------------------------------------------------
    # 1. helper functions
    # ---------------------------------------------------------
    @staticmethod
    def _wrap_pi(theta):
        """Wrap an angle to [-pi/2, pi/2)."""
        return (theta + np.pi / 2) % np.pi - np.pi / 2

    @staticmethod
    def _longedge_normalize(box):
        """Unify the long-edge representation on a copy: swap w/h and theta+=pi/2
        when h>w; wrap to [-pi/2, pi/2). Never modify inputs in place."""
        b = np.asarray(box, dtype=np.float64).copy()
        if b.shape[0] >= 5:
            w, h = b[2], b[3]
            if h > w:
                b[2], b[3] = h, w
                b[4] = b[4] + np.pi / 2
            b[4] = DOTAMetric._wrap_pi(b[4])
        return b

    @staticmethod
    def _aspect_ratio(box):
        b = np.asarray(box, dtype=np.float64)
        w, h = abs(b[2]), abs(b[3])
        return max(w, h) / min(w, h) if min(w, h) > 1e-6 else float('inf')

    def _calculate_mAngle_raw(self, det_box, gt_box):
        """Legacy angle deviation (without long-edge unification), in degrees."""
        a_det = det_box[4]
        a_gt = gt_box[4]
        diff = a_det - a_gt
        diff = (diff + np.pi / 2) % np.pi - np.pi / 2
        return abs(diff * 180 / np.pi)

    def _calculate_mAngle_longedge(self, det_box, gt_box):
        """Minimum angular difference modulo pi after long-edge unification (degrees)."""
        d = self._longedge_normalize(det_box)
        g = self._longedge_normalize(gt_box)
        diff = DOTAMetric._wrap_pi(d[4] - g[4])
        return abs(diff * 180 / np.pi)

    def _calculate_mSize(self, det_box, gt_box):
        """Size deviation: de-rotate to an axis-aligned box, align centers, compute horizontal IoU."""
        w_det, h_det = det_box[2], det_box[3]
        w_gt, h_gt = gt_box[2], gt_box[3]
        # axis-aligned box centered at (0, 0): [x1, y1, x2, y2]
        det_rect = torch.tensor([[-w_det/2, -h_det/2, w_det/2, h_det/2]])
        gt_rect = torch.tensor([[-w_gt/2, -h_gt/2, w_gt/2, h_gt/2]])
        iou = bbox_overlaps(det_rect, gt_rect, is_aligned=True)
        return iou.item()

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        collector = defaultdict(list)
        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5], cls_dets[:, -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8], cls_dets[:, -1], self.iou_thr)
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'{outfile_prefix} exists.')
        os.makedirs(outfile_prefix)

        files = [osp.join(outfile_prefix, 'Task1_' + cls + '.txt') for cls in self.dataset_meta['classes']]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0: continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                for qbox, score in zip(qboxes, scores):
                    txt_element = [img_id, str(round(float(score), 2))] + [f'{p:.2f}' for p in qbox]
                    f.writelines(' '.join(txt_element) + '\n')
        for f in file_objs: f.close()
        target_name = osp.split(outfile_prefix)[-1]
        zip_path = osp.join(outfile_prefix, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files: t.write(f, osp.split(f)[-1])
        return zip_path

    def results2json(self, results: Sequence[dict], outfile_prefix: str) -> dict:
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                data['category_id'] = int(label)
                bbox_json_results.append(data)
        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])
        return result_files

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape((-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)
            self.results.append((ann, result))

    # ---------------------------------------------------------
    # 2. main compute_metrics logic
    # ---------------------------------------------------------
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            first_iou_done = False
            
            # global accumulators
            total_tp_count = 0
            total_mAngle = 0.0
            total_mSize = 0.0
            total_mIoU = 0.0
            total_mAngle_le = 0.0
            total_mAngle_le_slim = 0.0
            total_slim_cnt = 0
            total_mAngle_le_nosq = 0.0
            total_tp_nosq = 0

            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                
                # call eval_rbbox_map to get cls_matched_pairs (with iou)
                mean_ap, eval_results_list, cls_matched_pairs = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger='silent')  # silent first; the merged table is printed later
                
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                
                # detailed stats only for the first IoU threshold (usually 0.5)
                if not first_iou_done:
                    for i, class_pairs in enumerate(cls_matched_pairs):
                        if len(class_pairs) > 0:
                            c_mAngle = 0.0
                            c_mSize = 0.0
                            c_mIoU = 0.0
                            c_mAngle_le = 0.0
                            c_mAngle_le_slim = 0.0
                            c_slim_cnt = 0
                            
                            # iterate over all TP triples (det, gt, rotated_iou)
                            for det_box, gt_box, rotated_iou in class_pairs:
                                c_mAngle += self._calculate_mAngle_raw(det_box, gt_box)
                                c_mSize += self._calculate_mSize(det_box, gt_box)
                                c_mIoU += rotated_iou  # accumulate the rotated IoU
                                c_mAngle_le += self._calculate_mAngle_longedge(det_box, gt_box)
                                # TP subset with aspect ratio >= 1.5
                                if (self._aspect_ratio(det_box) >= 1.5
                                        and self._aspect_ratio(gt_box) >= 1.5):
                                    c_mAngle_le_slim += self._calculate_mAngle_longedge(
                                        det_box, gt_box)
                                    c_slim_cnt += 1
                            
                            n = len(class_pairs)
                            # per-class results for print_map_summary
                            eval_results_list[i]['mAngle'] = c_mAngle / n  # keep the legacy column (raw)
                            eval_results_list[i]['mAngle_raw'] = c_mAngle / n
                            eval_results_list[i]['mAngle_longedge'] = c_mAngle_le / n
                            eval_results_list[i]['mAngle_le_cnt'] = n
                            eval_results_list[i]['mSize'] = c_mSize / n
                            eval_results_list[i]['mIoU'] = c_mIoU / n
                            if c_slim_cnt > 0:
                                eval_results_list[i]['mAngle_le_slim'] = c_mAngle_le_slim / c_slim_cnt
                                eval_results_list[i]['mAngle_le_slim_cnt'] = c_slim_cnt
                            else:
                                eval_results_list[i]['mAngle_le_slim'] = 'N/A'
                                eval_results_list[i]['mAngle_le_slim_cnt'] = 0
                            
                            # accumulate into the global mean
                            total_mAngle += c_mAngle
                            total_mSize += c_mSize
                            total_mIoU += c_mIoU
                            total_mAngle_le += c_mAngle_le
                            total_mAngle_le_slim += c_mAngle_le_slim
                            total_slim_cnt += c_slim_cnt
                            total_tp_count += n
                            # accumulate TP excluding direction-agnostic classes
                            if i not in self.square_cls:
                                total_mAngle_le_nosq += c_mAngle_le
                                total_tp_nosq += n
                        else:
                            eval_results_list[i]['mAngle'] = 'N/A'
                            eval_results_list[i]['mAngle_raw'] = 'N/A'
                            eval_results_list[i]['mAngle_longedge'] = 'N/A'
                            eval_results_list[i]['mAngle_le_cnt'] = 0
                            eval_results_list[i]['mAngle_le_slim'] = 'N/A'
                            eval_results_list[i]['mAngle_le_slim_cnt'] = 0
                            eval_results_list[i]['mSize'] = 'N/A'
                            eval_results_list[i]['mIoU'] = 'N/A'
                    
                    first_iou_done = True
                    
                    # call print_map_summary to print the table with new metrics
                    print_map_summary(
                        mean_ap, 
                        eval_results_list, 
                        dataset_name, 
                        self.scale_ranges, 
                        logger=logger
                    )

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
            
            # global mean stored into the returned dict
            if total_tp_count > 0:
                final_mAngle = total_mAngle / total_tp_count
                final_mSize = total_mSize / total_tp_count
                final_mIoU = total_mIoU / total_tp_count
                final_mAngle_le = total_mAngle_le / total_tp_count
                final_mAngle_le_slim = (total_mAngle_le_slim / total_slim_cnt
                                        if total_slim_cnt > 0 else 'N/A')
                final_mAngle_le_nosq = (total_mAngle_le_nosq / total_tp_nosq
                                        if total_tp_nosq > 0 else 'N/A')
            else:
                final_mAngle = final_mSize = final_mIoU = 0.0
                final_mAngle_le = 0.0
                final_mAngle_le_slim = 'N/A'
                final_mAngle_le_nosq = 'N/A'
            
            eval_results['mIoU'] = round(final_mIoU, 3)
            eval_results['mAngle_raw'] = round(final_mAngle, 3)
            eval_results['mAngle_longedge'] = round(final_mAngle_le, 3)
            eval_results['mSize'] = round(final_mSize, 3)
            eval_results['mAngle_le_cnt'] = total_tp_count
            if isinstance(final_mAngle_le_slim, float):
                eval_results['mAngle_le_slim'] = round(final_mAngle_le_slim, 3)
                eval_results['mAngle_le_slim_cnt'] = total_slim_cnt
            else:
                eval_results['mAngle_le_slim'] = 'N/A'
                eval_results['mAngle_le_slim_cnt'] = 0
            if isinstance(final_mAngle_le_nosq, float):
                eval_results['mAngle_longedge_nosq'] = round(final_mAngle_le_nosq, 3)
            eval_results['mAngle_tp_cnt'] = total_tp_count
            
        else:
            raise NotImplementedError
        return eval_results
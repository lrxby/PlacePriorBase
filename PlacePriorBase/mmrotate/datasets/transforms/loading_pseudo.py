# Copyright (c) OpenMMLab. All rights reserved.
"""Pseudo-label loading with instance-aligned matching and validity mask.

Fixes three bugs of the original loader:

1. **Zero-box placeholder supervision.**  When an image had no record in the
   pkl, the old loader generated ``num_gts`` all-zero boxes and fed them into
   ``loss_ourwater`` as valid supervision.  Now a zero-box placeholder with an
   all-``False`` validity mask is produced instead, so the boxes never
   contribute to ``VWRBoxLoss`` (classification / point supervision are
   untouched).

2. **Row-index misalignment.**  The pseudo boxes in the pkl are generated from
   the *full* ``annfiles`` while the training GT may be filtered later (e.g.
   ``ignore_flag`` when ``difficulty > diff_thr``).  The old code let the head
   index ``pseudo_boxes`` by the *filtered* GT row number, silently matching
   the wrong instance whenever GT and pkl counts differ (the danger is not
   only out-of-bounds: an in-bounds but wrong row, e.g. ``[A,B,C]`` with ``B``
   ignored gives GT ``[A,C]`` and ``pseudo_boxes[1]`` is ``B'``, not ``C'``).

   This loader re-aligns pseudo boxes to the *current* GT instances by
   category + intra-category order (both the pkl and the annfiles preserve the
   original row order, so GT is a subsequence of the pkl rows).  A GT instance
   whose category/order has no reliable match gets ``pseudo_valid=False`` and
   is excluded from ``VWRBoxLoss`` — never guessed.

3. **No stats.**  Aggregated counters (missing images, unmatched instances,
   invalid boxes) are kept and printed periodically instead of per-iter spam.
"""
import os
import pickle

import numpy as np
from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPseudoAnnotations:
    """Load offline pseudo rboxes and align them to the current GT instances.

    Args:
        pkl_path (str): Path to the pseudo-label pickle. The pickle must map
            ``image_id -> list of [cx, cy, w, h, angle, label]``.
    """

    # class-level aggregation so that many workers print the summary
    # occasionally instead of every iteration
    _stats = dict(total_images=0, missing_images=0, total_gt=0,
                  unmatched_gt=0, invalid_pseudo=0)

    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f'[LoadPseudoAnnotations] pseudo-label pkl not found: '
                f'{pkl_path}')
        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        print(f'[LoadPseudoAnnotations] loaded offline pseudo labels: '
              f'{pkl_path} ({len(self.data_dict)} images)')

    # ------------------------------------------------------------------
    # matching helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_row(row5):
        """A pseudo row [cx,cy,w,h,theta] is valid iff finite and w,h>0.

        Note: 16x16 fallback boxes are legitimate (the generator uses them as
        minimal boxes); they are NOT removed here.
        """
        if len(row5) != 5:
            return False
        cx, cy, w, h = row5[0], row5[1], row5[2], row5[3]
        if not (np.isfinite(row5).all()):
            return False
        if not (w > 0 and h > 0):
            return False
        _ = (cx, cy)
        return True

    def _align_to_gt(self, pseudo_rows, gt_labels, gt_centers, match_thresh=32.0):
        """Align pseudo rows to GT instances by category + nearest-center.

        The pkl generator anchors every pseudo box center EXACTLY at the GT
        point (``center = gt_points[idx]``), so GT centers and pseudo centers
        coincide up to floating-point error.  Nearest-neighbour matching in the
        same category therefore stays correct even when a same-category
        instance is filtered out by ignore (the "row-index is legal but the
        instance is wrong" case).  A GT instance whose nearest same-class
        pseudo row is farther than ``match_thresh`` is treated as having no
        reliable pseudo label (valid=False) — never guessed.
        Returns (pseudo_boxes[N,5], valid[N]).
        """
        n_gt = len(gt_labels)
        pseudo_boxes = np.zeros((n_gt, 5), dtype=np.float32)
        pseudo_valid = np.zeros(n_gt, dtype=bool)

        # group pseudo row indices by category, preserving row order
        pseudo_by_cls = {}
        for row_i, row in enumerate(pseudo_rows):
            lab = int(row[5]) if len(row) > 5 else -1
            pseudo_by_cls.setdefault(lab, []).append(row_i)

        matched = np.zeros(len(pseudo_rows), dtype=bool)
        for gt_i, lab in enumerate(gt_labels):
            lab = int(lab)
            lst = pseudo_by_cls.get(lab)
            if lst is None:
                continue  # no pseudo of this category -> invalid
            # nearest unmatched pseudo center in the same category
            best_i, best_d = None, float('inf')
            for row_i in lst:
                if matched[row_i]:
                    continue
                dx = pseudo_rows[row_i][0] - gt_centers[gt_i, 0]
                dy = pseudo_rows[row_i][1] - gt_centers[gt_i, 1]
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d, best_i = d, row_i
            if best_i is None or best_d > match_thresh * match_thresh:
                continue  # no reliable pseudo for this instance
            matched[best_i] = True
            row5 = np.asarray(pseudo_rows[best_i][:5], dtype=np.float32)
            if self._is_valid_row(row5):
                pseudo_boxes[gt_i] = row5
                pseudo_valid[gt_i] = True
            else:
                LoadPseudoAnnotations._stats['invalid_pseudo'] += 1
            # note: an unmatched but in-bounds pseudo row is NEVER guessed

        n_unmatched = n_gt - int(pseudo_valid.sum())
        LoadPseudoAnnotations._stats['unmatched_gt'] += n_unmatched
        return pseudo_boxes, pseudo_valid

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def __call__(self, results):
        file_name = results.get('img_path', '')
        if not file_name:
            file_name = results.get('img_info', {}).get('filename', '')
        file_id = os.path.splitext(os.path.basename(file_name))[0]

        gt_labels = results.get('gt_bboxes_labels', None)
        n_gt = len(gt_labels) if gt_labels is not None else 0
        gt_labels_np = np.asarray(
            gt_labels, dtype=np.int64) if gt_labels is not None else np.zeros(
                0, dtype=np.int64)

        # GT center (mean of the 8 qbox points) for nearest-neighbor matching
        gt_centers = np.zeros((n_gt, 2), dtype=np.float32)
        gt_bboxes = results.get('gt_bboxes', None)
        if gt_bboxes is not None and n_gt > 0:
            if hasattr(gt_bboxes, 'tensor'):
                gtb = gt_bboxes.tensor
            else:
                gtb = gt_bboxes
            if hasattr(gtb, 'numpy'):
                gtb = gtb.detach().cpu().numpy() if hasattr(gtb, 'detach') else gtb.numpy()
            gtb = np.asarray(gtb, dtype=np.float32).reshape(n_gt, -1)
            if gtb.shape[1] >= 8:
                gt_centers = gtb.reshape(n_gt, 4, 2).mean(axis=1)
            elif gtb.shape[1] == 2:
                gt_centers = gtb
        # matching threshold: 3% of the image size (min 32px)
        ori_shape = results.get('ori_shape', None) or results.get('img_shape', None)
        if ori_shape is not None and len(ori_shape) >= 2:
            match_thresh = max(32.0, 0.03 * max(float(ori_shape[0]), float(ori_shape[1])))
        else:
            match_thresh = 32.0

        LoadPseudoAnnotations._stats['total_images'] += 1
        LoadPseudoAnnotations._stats['total_gt'] += n_gt

        if file_id in self.data_dict:
            pseudo_rows = self.data_dict[file_id]
            if len(pseudo_rows) > 0 and n_gt > 0:
                pseudo_boxes, pseudo_valid = self._align_to_gt(
                    pseudo_rows, gt_labels_np, gt_centers, match_thresh)
            elif len(pseudo_rows) > 0 and n_gt == 0:
                # GT empty: nothing to supervise
                pseudo_boxes = np.zeros((0, 5), dtype=np.float32)
                pseudo_valid = np.zeros(0, dtype=bool)
            else:
                # empty record: placeholder aligned to GT, all invalid
                pseudo_boxes = np.zeros((n_gt, 5), dtype=np.float32)
                pseudo_valid = np.zeros(n_gt, dtype=bool)
        else:
            # image has no record: placeholder + all-False mask
            LoadPseudoAnnotations._stats['missing_images'] += 1
            pseudo_boxes = np.zeros((n_gt, 5), dtype=np.float32)
            pseudo_valid = np.zeros(n_gt, dtype=bool)

        results['pseudo_boxes'] = pseudo_boxes
        results['pseudo_valid'] = pseudo_valid

        if 'bbox_fields' not in results:
            results['bbox_fields'] = []
        if 'pseudo_boxes' not in results['bbox_fields']:
            results['bbox_fields'].append('pseudo_boxes')

        # periodic summary (every 300 images per worker, avoids log spam)
        s = LoadPseudoAnnotations._stats
        if s['total_images'] % 300 == 0:
            print(f'[LoadPseudoAnnotations] stats: images={s["total_images"]} '
                  f'missing={s["missing_images"]} gt={s["total_gt"]} '
                  f'unmatched_gt={s["unmatched_gt"]} '
                  f'invalid_pseudo={s["invalid_pseudo"]}')
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(pkl_path={self.pkl_path})'

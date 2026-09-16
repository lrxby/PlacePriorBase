# Copyright (c) OpenMMLab. All rights reserved.
import os, copy, math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale, ConvModule
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, unpack_gt_instances)
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, rbox2qbox, hbox2rbox, rbox2hbox

INF = 1e8


@MODELS.register_module()
class PlacePriorBaseHead(AnchorFreeHead):
    """
    Simplified PlacePriorBaseHead with Offline Pseudo-Label Supervision.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 strides: list = [8],
                 regress_ranges: list = [(-1, 1e8)],
                 center_sampling: bool = True,
                 center_sample_radius: float = 0.75,
                 angle_version: str = 'le90',
                 pseudo_generator: bool = False,
                 voronoi_type: str = 'gaussian-orientation',
                 voronoi_thres: dict = dict(default=[0.994, 0.005]),
                 square_cls: list = [],
                 post_process: dict = {},
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 angle_coder: ConfigType = dict(
                    type='PSCCoder',
                    angle_version='le90',
                    dual_freq=False,
                    num_step=3,
                    thr_mod=0),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 
                 # --- Losses ---
                 loss_bbox: ConfigType = dict(
                     type='GDLoss', 
                     loss_type='gwd', 
                     loss_weight=5.0),
                 loss_voronoi: ConfigType = dict(
                     type='VoronoiWatershedLoss', loss_weight=5.0),
                 loss_size: ConfigType = dict(
                     type='SizePriorLoss', loss_weight=1.0),
                 loss_angle: ConfigType = dict(
                     type='AnglePriorLoss', loss_weight=1.0),
                 loss_ourwater: ConfigType = dict(  
                     type='VWRBoxLoss',
                     loss_weight=1.0),
                 
                 # --- Loss Switches ---
                 use_bbox_loss: bool = True,
                 use_size_loss: bool = True,
                 use_angle_loss: bool = True,
                 use_voronoi_loss: bool = True,   
                 use_ourwater_loss: bool = False, 
                 
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='conv_cls',
                             std=0.01,
                             bias_prob=0.01), 
                         dict(
                             type='Normal',
                             name='conv_gate',
                             std=0.01,
                             bias_prob=0.01)]),
                 **kwargs):
        
        self.angle_coder = TASK_UTILS.build(angle_coder)
        
        super().__init__(
            num_classes,
            in_channels,
            strides=strides,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox, 
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
            
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.angle_version = angle_version
        self.pseudo_generator = pseudo_generator
        self.voronoi_type = voronoi_type
        self.voronoi_thres = voronoi_thres
        self.square_cls = square_cls
        self.post_process = post_process
        
        self.use_bbox_loss = use_bbox_loss
        self.use_size_loss = use_size_loss
        self.use_angle_loss = use_angle_loss
        self.use_voronoi_loss = use_voronoi_loss 
        self.use_ourwater_loss = use_ourwater_loss 
        
        self.loss_voronoi = MODELS.build(loss_voronoi)
        self.loss_size = MODELS.build(loss_size)
        self.loss_angle = MODELS.build(loss_angle)
        self.loss_ourwater = MODELS.build(loss_ourwater)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_angle = nn.Conv2d(
            self.feat_channels, self.angle_coder.encode_size, 3, padding=1)
        self.conv_gate = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        
    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network."""
        cls_feat = x[0]
        reg_feat = x[0]

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        angle_pred = self.conv_angle(reg_feat)

        # Gaussian parameters: sig_x, sig_y, dx, dy
        sig_x = bbox_pred[:, 0].exp()
        sig_y = bbox_pred[:, 1].exp()
        dx = bbox_pred[:, 2].sigmoid() * 2 - 1  # (-1, 1)
        dy = bbox_pred[:, 3].sigmoid() * 2 - 1  # (-1, 1)
        
        bbox_pred = torch.stack((sig_x, sig_y, dx, dy), 1) * 8

        return (cls_score,), (bbox_pred,), (angle_pred,)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """
        Rewrite the parent loss method to explicitly pass batch_data_samples
        to loss_by_feat and fix the unpacking error.
        """
        outputs = self(x)
        
        # unpack 3 return values
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        
        return self.loss_by_feat(
            *outputs, 
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            batch_data_samples=batch_data_samples)  # passed explicitly

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        batch_data_samples: SampleList = None  # incoming samples
    ) -> Dict[str, Tensor]:
        """Calculate loss."""
        
        # --- Step 0: Extract Pseudo Labels ---
        batch_pseudo_boxes = []
        batch_pseudo_valid = []
        if batch_data_samples is not None:
            for data_sample in batch_data_samples:
                if hasattr(data_sample, 'pseudo_boxes'):
                    p_box = data_sample.pseudo_boxes
                    if isinstance(p_box, torch.Tensor):
                        batch_pseudo_boxes.append(p_box)
                    else:
                        batch_pseudo_boxes.append(torch.tensor(p_box, device=bbox_preds[0].device))
                    # pseudo_valid: boolean mask aligned with GT (loader aligns by class/order)
                    if hasattr(data_sample, 'pseudo_valid'):
                        pv = data_sample.pseudo_valid
                        if isinstance(pv, torch.Tensor):
                            batch_pseudo_valid.append(pv.to(device=bbox_preds[0].device))
                        else:
                            batch_pseudo_valid.append(
                                torch.tensor(pv, dtype=torch.bool, device=bbox_preds[0].device))
                    else:
                        # no mask (old pkl path): all valid by default
                        n = p_box.shape[0] if hasattr(p_box, 'shape') else len(p_box)
                        batch_pseudo_valid.append(
                            torch.ones(n, dtype=torch.bool, device=bbox_preds[0].device))
                else:
                    batch_pseudo_boxes.append(None)
                    batch_pseudo_valid.append(None)
        
        # --- Step 1: Pre-processing (Anchors/Targets) ---
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        # receive the 5 return values of get_targets, including gt_inds
        labels, bbox_targets, bid_targets, pseudo_targets, pseudo_valids, gt_inds = self.get_targets(
            all_level_points, batch_gt_instances, batch_pseudo_boxes=batch_pseudo_boxes,
            batch_pseudo_valid=batch_pseudo_valid)

        num_imgs = cls_scores[0].size(0)
        
        # Flatten tensors
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bid_targets = torch.cat(bid_targets)
        # flatten pseudo targets
        flatten_pseudo_targets = torch.cat(pseudo_targets)
        flatten_pseudo_valids = torch.cat(pseudo_valids)
        # flatten GT inds
        flatten_gt_inds = torch.cat(gt_inds)
        
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # --- Step 2: Classification Loss ---
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        # Select positive samples
        pos_cls_scores = flatten_cls_scores[pos_inds].sigmoid()
        pos_labels = flatten_labels[pos_inds]
        pos_cls_scores = torch.gather(pos_cls_scores, 1, pos_labels[:, None])[:, 0]

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]
        
        # pseudo labels for positive samples
        pos_pseudo_targets = flatten_pseudo_targets[pos_inds]
        pos_pseudo_valids = flatten_pseudo_valids[pos_inds]
        # GT IDs of positive samples
        pos_gt_ids = flatten_gt_inds[pos_inds]

        # [Initialize Losses]
        loss_voronoi = pos_bbox_preds.new_tensor(0.0)
        val_loss_bbox = pos_bbox_preds.new_tensor(0.0) 
        val_loss_size = pos_bbox_preds.new_tensor(0.0)
        val_loss_angle = pos_bbox_preds.new_tensor(0.0)
        val_loss_ourwater = pos_bbox_preds.new_tensor(0.0)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            
            # --- Step 3: Geometry Construction ---
            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)
            
            square_mask = torch.zeros_like(pos_labels, dtype=torch.bool)
            for c in self.square_cls:
                square_mask = torch.logical_or(square_mask, pos_labels == c)
            pos_decoded_angle_preds[square_mask] = 0

            # Decode targets
            pos_rbox_targets = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            
            # Construct prediction boxes
            pos_rbox_preds = torch.cat((pos_points + pos_bbox_preds[:, 2:], 
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            cos_r = torch.cos(pos_decoded_angle_preds)
            sin_r = torch.sin(pos_decoded_angle_preds)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            pos_gaus_preds = R.matmul(torch.diag_embed(pos_bbox_preds[:, :2])).matmul(R.permute(0, 2, 1))

            # [loss_bbox Logic]
            pos_syn_mask = pos_bid_targets[:, 1] == 1 
            pos_rbox_targets[~pos_syn_mask, 2:] = pos_rbox_preds[~pos_syn_mask, 2:].detach()
            
            if self.use_bbox_loss:
                val_loss_bbox = self.loss_bbox(
                    pos_rbox_preds,
                    pos_rbox_targets,
                    avg_factor=num_pos)
            
            # Use gt point to replace predicted center for downstream losses
            pos_rbox_preds = torch.cat((pos_rbox_targets[:, :2], 
                                        pos_bbox_preds[:, :2] * 2,
                                        pos_decoded_angle_preds), -1)

            # --- Step 4: Grouping ---
            bid_with_view = pos_bid_targets[:, 3] + 0.5 * pos_bid_targets[:, 2]
            bid, idx = torch.unique(bid_with_view, return_inverse=True)
            
            ins_bids = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 3], 'amin', include_self=False)
            ins_batch = pos_bid_targets.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_bid_targets[:, 0], 'amin', include_self=False)
            ins_labels = pos_labels.new_zeros(*bid.shape).index_reduce_(
                    0, idx, pos_labels, 'amin', include_self=False)
            
            ins_gaus_preds = pos_gaus_preds.new_zeros(
                *bid.shape, 4).index_reduce_(
                    0, idx, pos_gaus_preds.view(-1, 4), 'mean',
                    include_self=False).view(-1, 2, 2)
            
            ins_rbox_targets = pos_rbox_targets.new_zeros(
                *bid.shape, pos_rbox_targets.shape[-1]).index_reduce_(
                    0, idx, pos_rbox_targets, 'mean',
                    include_self=False)

            ori_mu_all = ins_rbox_targets[:, 0:2]

            # --- Step 5: Voronoi Loss ---
            if self.use_voronoi_loss:
                for batch_id in range(len(batch_gt_instances)):
                    group_mask = (ins_batch == batch_id) & (ins_bids != 0)
                    mu = ori_mu_all[group_mask]
                    sigma = ins_gaus_preds[group_mask]
                    label = ins_labels[group_mask]
                    
                    if len(mu) >= 1:
                        pos_thres = [self.voronoi_thres['default'][0]] * self.num_classes
                        neg_thres = [self.voronoi_thres['default'][1]] * self.num_classes
                        if 'override' in self.voronoi_thres.keys():
                            for item in self.voronoi_thres['override']:
                                for cls in item[0]:
                                    pos_thres[cls] = item[1][0]
                                    neg_thres[cls] = item[1][1]
                        
                        loss_voronoi += self.loss_voronoi(
                            (mu, sigma.bmm(sigma)),
                            label, self.images[batch_id],
                            pos_thres, neg_thres,
                            voronoi=self.voronoi_type)

                loss_voronoi = loss_voronoi / len(batch_gt_instances)

            # --- Step 6: Other Losses ---
            pos_batch_idxs = pos_bid_targets[:, 0]

            if self.use_size_loss:
                val_loss_size = self.loss_size(
                    pred_bboxes=pos_rbox_preds,
                    scores=pos_cls_scores,
                    labels=pos_labels,
                    batch_idxs=pos_batch_idxs,
                    img_metas=batch_img_metas
                )

            if self.use_angle_loss:
                val_loss_angle = self.loss_angle(
                    pos_bbox_preds=pos_rbox_preds,
                    pos_scores=pos_cls_scores,
                    pos_labels=pos_labels,
                    batch_idxs=pos_batch_idxs,
                    pos_gt_ids=pos_gt_ids  # pass positive GT IDs
                )
            
            # VWRBoxLoss: supervise only instances with valid pseudo labels
            if self.use_ourwater_loss:
                valid_mask = pos_pseudo_valids
                if valid_mask.any():
                    val_loss_ourwater = self.loss_ourwater(
                        pos_rbox_preds[valid_mask],
                        pos_pseudo_targets[valid_mask].detach()
                    )
                else:
                    # no valid pseudo labels: return zero loss on the graph
                    val_loss_ourwater = pos_bbox_preds.sum() * 0

        losses = dict(loss_cls=loss_cls)

        if self.use_voronoi_loss:
            losses['loss_bbox_vor'] = loss_voronoi

        if self.use_bbox_loss:
            losses['loss_bbox'] = val_loss_bbox

        if self.use_size_loss:
            losses['loss_size'] = val_loss_size
            
        if self.use_angle_loss:
            losses['loss_angle'] = val_loss_angle
            
        if self.use_ourwater_loss:
            losses['loss_ourwater'] = val_loss_ourwater

        return losses

    def get_targets(
        self, 
        points: List[Tensor], 
        batch_gt_instances: InstanceList,
        batch_pseudo_boxes: List[Tensor] = None,
        batch_pseudo_valid: List[Tensor] = None
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:  # extended return annotation
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]
        
        # keep list lengths aligned for multi_apply
        if batch_pseudo_boxes is None or len(batch_pseudo_boxes) == 0:
            batch_pseudo_boxes = [None] * len(batch_gt_instances)
        if batch_pseudo_valid is None or len(batch_pseudo_valid) == 0:
            batch_pseudo_valid = [None] * len(batch_gt_instances)

        # receive the 5th/6th returns: gt_inds_list / pseudo_valid_list
        labels_list, bbox_targets_list, bid_targets_list, pseudo_targets_list, pseudo_valid_list, gt_inds_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            batch_pseudo_boxes,
            batch_pseudo_valid,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bid_targets_list = [
            bid_targets.split(num_points, 0)
            for bid_targets in bid_targets_list
        ]
        pseudo_targets_list = [
            pseudo_targets.split(num_points, 0)
            for pseudo_targets in pseudo_targets_list
        ]
        pseudo_valid_list = [
            pseudo_valid.split(num_points, 0)
            for pseudo_valid in pseudo_valid_list
        ]
        # split gt_inds_list
        gt_inds_list = [
            gt_inds.split(num_points, 0)
            for gt_inds in gt_inds_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bid_targets = []
        concat_lvl_pseudo_targets = [] 
        concat_lvl_pseudo_valid = []
        concat_lvl_gt_inds = []
        
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bid_targets = torch.cat(
                [bid_targets[i] for bid_targets in bid_targets_list])
            pseudo_targets = torch.cat(
                [pseudo_targets[i] for pseudo_targets in pseudo_targets_list])
            pseudo_valid = torch.cat(
                [pseudo_valid[i] for pseudo_valid in pseudo_valid_list])
            # concatenate gt_inds
            gt_inds = torch.cat(
                [gt_inds[i] for gt_inds in gt_inds_list])
                
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bid_targets.append(bid_targets)
            concat_lvl_pseudo_targets.append(pseudo_targets)
            concat_lvl_pseudo_valid.append(pseudo_valid)
            concat_lvl_gt_inds.append(gt_inds)
            
        # return gt_inds and pseudo_valid
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_bid_targets, concat_lvl_pseudo_targets,
                concat_lvl_pseudo_valid, concat_lvl_gt_inds)

    def _get_targets_single(
            self, 
            gt_instances: InstanceData, 
            pseudo_boxes: Optional[Tensor], 
            pseudo_valid: Optional[Tensor],
            points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # type annotation
        
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bids = gt_instances.bids

        if num_gts == 0:
            # return empty gt_inds and pseudo_valid
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 5)), \
                   gt_bids.new_zeros((num_points, 4)), \
                   points.new_zeros((num_points, 5)), \
                   points.new_zeros((num_points,), dtype=torch.bool), \
                   gt_labels.new_full((num_points,), -1, dtype=torch.long)  # GT ID defaults to -1

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.tensor

        # ... (code unchanged) ...
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)
        
        offset = points - gt_ctr
        w, h = gt_wh[..., 0].clone(), gt_wh[..., 1].clone()

        center_r = torch.clamp((w * h).sqrt() / 64, 1, 5)[..., None]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            inside_gt_bbox_mask = (abs(offset) < stride * center_r).all(dim=-1)

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes 
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bids[min_area_inds]
        
        # angle is concatenated, so dim=1 becomes 4+1=5
        bbox_targets = torch.cat((bbox_targets, angle_targets), -1)
        
        # GT inds of matched positive samples
        gt_inds = gt_labels.new_full((num_points,), -1, dtype=torch.long)
        # only points assigned to positives get real GT IDs
        # min_area_inds is the relative GT index (0..num_gts-1)
        valid_pos_mask = (min_area != INF)
        gt_inds[valid_pos_mask] = min_area_inds[valid_pos_mask]
        
        # Pseudo target assignment: do not silently clamp mismatches.
        # pseudo_boxes are aligned with GT by the loader (len == num_gts).
        # Out-of-range indices mean no valid pseudo box (valid=False).
        point_pseudo_targets = points.new_zeros((num_points, 5))
        point_pseudo_valid = points.new_zeros((num_points,), dtype=torch.bool)
        if pseudo_boxes is not None and len(pseudo_boxes) > 0:
            # Mask for valid assignments (not background)
            valid_mask = (min_area != INF)
            
            # relative GT index (0..num_gts-1)
            valid_inds = min_area_inds[valid_mask]
            # out-of-range instances have no valid pseudo box
            in_bounds = valid_inds < len(pseudo_boxes)
            # valid mask: check bounds, then combine with the loader pseudo_valid
            ok = in_bounds.clone()
            if pseudo_valid is not None and len(pseudo_valid) > 0:
                pv = pseudo_valid
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv, dtype=torch.bool, device=points.device)
                else:
                    pv = pv.to(device=points.device)
                if len(pv) > 0:
                    ok = ok.clone()
                    ok[in_bounds] = pv[valid_inds[in_bounds].clamp(max=len(pv) - 1)]
            # assign real pseudo boxes only to valid points
            idx = valid_inds[ok]
            assign_pos = torch.zeros(num_points, dtype=torch.bool, device=points.device)
            assign_pos[valid_mask] = ok
            point_pseudo_targets[assign_pos] = pseudo_boxes[idx]
            point_pseudo_valid[valid_mask] = ok

        # return gt_inds and pseudo_valid
        return labels, bbox_targets, bid_targets, point_pseudo_targets, point_pseudo_valid, gt_inds

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict detection results."""
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_data_samples=unpack_gt_instances(batch_data_samples), rescale=rescale)
        return predictions
    
    def predict_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            batch_data_samples: Optional[List[dict]] = None,
            cfg: Optional[ConfigDict] = None,
            rescale: bool = False,
            with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into bbox results."""
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []
        for img_id in range(len(batch_data_samples[2])):
            data_sample = (batch_data_samples[0][img_id], batch_data_samples[1][img_id], batch_data_samples[2][img_id])
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            
            if self.training or self.pseudo_generator:
                predict_by_feat = self._predict_by_feat_single_pseudo
            else:
                predict_by_feat = self._predict_by_feat_single

            results = predict_by_feat(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                mlvl_priors=mlvl_priors,
                data_sample=data_sample,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list
    
    def _predict_by_feat_single_pseudo(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                data_sample: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox results (Pseudo mode)."""        
        gt_instances = data_sample[0]
        img_meta = data_sample[2]
        if self.training:
            scale_factor = [1, 1]
        else:
            scale_factor = img_meta['scale_factor']
        gt_bboxes = gt_instances.bboxes.tensor
        gt_labels = gt_instances.labels
        gt_pos = (gt_bboxes[:, 0:2] / self.strides[0] * scale_factor[1]).long()

        cls_score, bbox_pred, angle_pred = cls_score_list[0], bbox_pred_list[0], angle_pred_list[0]
        H, W = cls_score.shape[1:3]

        gt_valid_mask = (0 <= gt_pos[:, 0]) & (gt_pos[:, 0] < W) & (0 <= gt_pos[:, 1]) & (gt_pos[:, 1] < H)
        gt_idx = gt_pos[:, 1] * W + gt_pos[:, 0]
        gt_idx = gt_idx.clamp(0, cls_score[0].numel() - 1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)[gt_idx]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)[gt_idx]

        angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)[gt_idx]
        decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
        bboxes = torch.cat((gt_bboxes[:, 0:2], bbox_pred[:, :2] * 2, decoded_angle), -1)

        bboxes[~gt_valid_mask, 2:] = 0
        bboxes[:, 2:4] = bboxes[:, 2:4] / scale_factor[1]

        for id in self.post_process.keys():
            bboxes[gt_labels == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[gt_labels == id, -1] = 0

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes.detach())
        results.scores = torch.ones_like(cls_score[:, 0])
        results.labels = gt_labels

        return results

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                data_sample: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into bbox results (Normal mode)."""
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_meta = data_sample[2]
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (
                cls_score, bbox_pred, angle_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat((priors + bbox_pred[:, 2:], bbox_pred[:, :2] * 2, decoded_angle), -1)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        scores = torch.cat(mlvl_scores)
        labels = torch.cat(mlvl_labels)
        bboxes = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)

        for id in self.post_process.keys():
            bboxes[labels == id, 2:4] *= self.post_process[id]
        for id in self.square_cls:
            bboxes[labels == id, -1] = 0
        
        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = scores
        results.labels = labels
        
        results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        return results
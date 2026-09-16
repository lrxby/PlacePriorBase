# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
from torch import Tensor

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmrotate.registry import MODELS


@MODELS.register_module()
class PlacePriorBase(SingleStageDetector):
    """
    PlacePriorBase-v2 (single-stream version).

    Compared with the full version, the following modules are removed to
    return to pure geometric constraints:
    1. Consistency learning removed: no dual-stream rotated/flipped images.
    2. TED (Token-based Edge Detector) removed: no extra edge network.
    3. Copy-Paste augmentation removed: simplified data flow.

    Core mechanism preserved: batch IDs and raw images are injected manually
    so that the head can compute Voronoi and geometric losses.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # TED initialization removed
        # consistency-loss weight removed

    def set_epoch(self, epoch):
        """
        Sync the epoch to the bbox head.
        Kept for future extension or enabling certain losses at specific epochs.
        """
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """
        Compute losses.
        """
        # 1. unpack ground truths (bboxes and labels)
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)

        # 2. assign instance IDs (bids)
        # The head groups instances of the same image via `bids`.
        # bids format: [batch_idx, syn_flag, view_idx, instance_idx]
        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances.bboxes)
            # bids tensor (N, 4)
            bids = gt_instances.labels.new_zeros(blen, 4)
            
            # Col 0: batch index (distinguishes images)
            bids[:, 0] = i          
            
            # Col 1: synthesis flag (0=real, 1=copy-paste); all real here
            # bids[:, 1] = 0        
            
            # Col 2: view index (0=original, 1=augmented); single stream only
            # bids[:, 2] = 0        
            
            # Col 3: unique instance ID (global, prevents cross-image confusion)
            bids[:, 3] = torch.arange(0, blen, 1) + offset 
            
            # attach bids to gt_instances; the head reads them
            gt_instances.bids = bids
            offset += blen

        # rotate_crop / flip augmentation removed
        # copy-paste augmentation removed
        # TED edge extraction removed

        # 3. inject raw images into the head
        # The loss needs raw pixels for the energy map.
        # Standard detectors do not pass images to the head; assign manually.
        self.bbox_head.images = batch_inputs

        # 4. extract features (backbone + neck)
        # a single forward pass over the batch
        x = self.extract_feat(batch_inputs)

        # 5. sync the modified GT (bids) back to batch_data_samples
        # The head loss receives batch_data_samples,
        # so bids must be updated in step 2.
        for data_sample, gt_instances in zip(batch_data_samples, batch_gt_instances):
            data_sample.gt_instances = gt_instances

        # 6. forward to compute losses
        losses = self.bbox_head.loss(x, batch_data_samples)

        return losses
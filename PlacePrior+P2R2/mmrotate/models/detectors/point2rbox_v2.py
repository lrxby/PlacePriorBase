# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import cv2
import numpy as np
from typing import Tuple, Union, List, Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import grid_sample
from torchvision import transforms

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import get_box_tensor
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes, rbox2hbox, hbox2rbox

from third_parties.ted.ted import TED


def get_single_pattern(image, bbox, label, square_cls):
    if bbox[2] < 16 or bbox[3] < 16 or bbox[2] > 512 or bbox[3] > 512:
        raise

    def obb2poly(obb):
        cx, cy, w, h, t = obb
        dw, dh = (w - 1) / 2, (h - 1) / 2
        cost = np.cos(t)
        sint = np.sin(t)
        mrot = np.float32([[cost, -sint], [sint, cost]])
        poly = np.float32([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
        return np.matmul(poly, mrot.T) + np.float32([cx, cy])

    def get_pattern_gaussian(w, h, device):
        w, h = int(w), int(h)
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij')
        y = (y - h / 2) / (h / 2)
        x = (x - w / 2) / (w / 2)
        ox, oy = torch.randn(2, device=device).clip(-3, 3) * 0.15
        sx, sy = torch.rand(2, device=device) * 0.5 + 1
        z = torch.exp(-((x - ox) * sx)**2 - ((y - oy) * sy)**2) * 0.5 + 0.5
        return z

    cx, cy, w, h, t = bbox
    w, h = int(w), int(h)
    poly = obb2poly([cx, cy, w, h, t])

    pts1 = poly[0:3]
    pts2 = np.float32([[-1, -1], [1, -1], [1, 1]])
    M = cv2.getAffineTransform(pts1, pts2)
    M = np.concatenate((M, ((0, 0, 1),)), 0)

    H, W = image.shape[1:3]
    T = np.array([[2 / W, 0, -1],
                  [0, 2 / H, -1],
                  [0, 0, 1]])
    theta = T @ np.linalg.inv(M)
    theta = image.new_tensor(theta[:2, :])[None]
    grid = F.affine_grid(theta, [1, 3, h, w], align_corners=True)
    chip = F.grid_sample(image[None], grid, align_corners=True)[0]

    alpha = get_pattern_gaussian(chip.shape[-1], chip.shape[-2], chip.device)[None]
    chip = torch.cat((chip, alpha))
        
    w, h, t = chip.new_tensor((bbox[2] * (0.7 + 0.5 * np.random.rand()), bbox[3] * (0.7 + 0.5 * np.random.rand()), np.pi * np.random.rand()))
    if label in square_cls:
        t *= 0
    cosa = torch.abs(torch.cos(t))
    sina = torch.abs(torch.sin(t))
    sx, sy = int(torch.ceil(cosa * w + sina * h)), int(torch.ceil(sina * w + cosa * h))
    theta = chip.new_tensor(
        [[1 / w * torch.cos(t), 1 / w * torch.sin(t), 0],
        [1 / h * torch.sin(-t), 1 / h * torch.cos(t), 0]])
    theta[:, :2] @= chip.new_tensor([[sx, 0], [0, sy]])
    grid = torch.nn.functional.affine_grid(
        theta[None], (1, 1, sy, sx), align_corners=True)
    chip = torch.nn.functional.grid_sample(
        chip[None], grid, align_corners=True, mode='nearest')[0]
    bbox = np.float32([sx / 2, sy / 2, w.item(), h.item(), t.item()])
    return (chip, bbox, label)


def get_copy_paste_cache(images, bboxes, labels, square_cls, num_copies):
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()
    patterns = []
    for b, l in zip(bboxes, labels):
        try:
            p = get_single_pattern(images, b, l, square_cls)
            patterns.append(p)
            if len(patterns) > num_copies:
                break
        except:
            pass
    return patterns


@MODELS.register_module()
class Point2RBoxV2(SingleStageDetector):
    """Implementation of `H2RBox-v2 <https://arxiv.org/abs/2304.04403>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 rotate_range: Tuple[float, float] = (0.25, 0.75),
                 scale_range: Tuple[float, float] = (0.5, 0.9),
                 ss_prob: float = [0.6, 0.15, 0.25],
                 copy_paste_start_epoch: int = 6,
                 num_copies: int = 10,
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

        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.ss_prob = ss_prob
        self.copy_paste_start_epoch = copy_paste_start_epoch
        self.num_copies = num_copies
        self.copy_paste_cache = None

        self.ted_model = TED()
        for param in self.ted_model.parameters():
            param.requires_grad = False
        self.ted_model.load_state_dict(torch.load('third_parties/ted/ted.pth'))
        self.ted_model.eval()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.bbox_head.epoch = epoch

    def rotate_crop(
            self,
            batch_inputs: Tensor,
            rot: float = 0.,
            size: Tuple[int, int] = (768, 768),
            batch_gt_instances: InstanceList = None,
            batch_pseudo_boxes: Optional[List[RotatedBoxes]] = None,
            padding: str = 'reflection'):
        device = batch_inputs.device
        n, c, h, w = batch_inputs.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        
        if rot != 0:
            cosa, sina = math.cos(rot), math.sin(rot)
            tf = batch_inputs.new_tensor([[cosa, -sina], [sina, cosa]],
                                         dtype=torch.float)
            x_range = torch.linspace(-1, 1, w, device=device)
            y_range = torch.linspace(-1, 1, h, device=device)
            y, x = torch.meshgrid(y_range, x_range)
            grid = torch.stack([x, y], -1).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            batch_inputs = grid_sample(
                batch_inputs, grid, 'bilinear', padding, align_corners=True)
                
            if batch_gt_instances is not None:
                for i, gt_instances in enumerate(batch_gt_instances):
                    gt_bboxes = get_box_tensor(gt_instances.bboxes)
                    xy, wh, a = gt_bboxes[..., :2], gt_bboxes[..., 2:4], gt_bboxes[..., [4]]
                    ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + rot
                    rot_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                    batch_gt_instances[i].bboxes = RotatedBoxes(rot_gt_bboxes)
                    
                    if batch_pseudo_boxes is not None and batch_pseudo_boxes[i] is not None:
                        pb_tensor = batch_pseudo_boxes[i].tensor
                        pxy, pwh, pa = pb_tensor[..., :2], pb_tensor[..., 2:4], pb_tensor[..., [4]]
                        pxy = (pxy - ctr).matmul(tf.T) + ctr
                        pa = pa + rot
                        batch_pseudo_boxes[i] = RotatedBoxes(torch.cat([pxy, pwh, pa], dim=-1))

        batch_inputs = batch_inputs[..., crop_h:crop_h + size_h, crop_w:crop_w + size_w]
        
        if batch_gt_instances is None:
            if batch_pseudo_boxes is not None:
                return batch_inputs, None, batch_pseudo_boxes
            return batch_inputs
        else:
            for i, gt_instances in enumerate(batch_gt_instances):
                gt_bboxes = get_box_tensor(gt_instances.bboxes)
                xy, wh, a = gt_bboxes[..., :2], gt_bboxes[..., 2:4], gt_bboxes[..., [4]]
                xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes = torch.cat([xy, wh, a], dim=-1)
                batch_gt_instances[i].bboxes = RotatedBoxes(crop_gt_bboxes)
                
                if batch_pseudo_boxes is not None and batch_pseudo_boxes[i] is not None:
                    pb_tensor = batch_pseudo_boxes[i].tensor
                    pxy, pwh, pa = pb_tensor[..., :2], pb_tensor[..., 2:4], pb_tensor[..., [4]]
                    pxy = pxy - pxy.new_tensor([[crop_w, crop_h]])
                    batch_pseudo_boxes[i] = RotatedBoxes(torch.cat([pxy, pwh, pa], dim=-1))

            if batch_pseudo_boxes is not None:
                return batch_inputs, batch_gt_instances, batch_pseudo_boxes
            return batch_inputs, batch_gt_instances
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        H, W = batch_inputs.shape[2:4]
        batch_gt_instances, _, batch_img_metas = unpack_gt_instances(batch_data_samples)

       # -------------------------------------------------------------
        # -------------------------------------------------------------
        batch_pseudo_boxes = []
        for ds, meta in zip(batch_data_samples, batch_img_metas):
            pb = getattr(ds, 'pseudo_boxes', None)
            if pb is None and 'pseudo_boxes' in meta:
                pb = meta['pseudo_boxes']
            
            if pb is not None:
                if not isinstance(pb, torch.Tensor):
                    pb = torch.tensor(pb, dtype=torch.float32)
                
                pb = pb.to(device=batch_inputs.device, dtype=torch.float32)
                
                if pb.size(-1) > 5:
                    pb = pb[..., :5]
                    
                batch_pseudo_boxes.append(RotatedBoxes(pb))
            else:
                batch_pseudo_boxes.append(None)
        # -------------------------------------------------------------

        offset = 1
        for i, gt_instances in enumerate(batch_gt_instances):
            blen = len(gt_instances.bboxes)
            bids = gt_instances.labels.new_zeros(blen, 4)
            bids[:, 0] = i
            bids[:, 3] = torch.arange(0, blen, 1) + offset
            gt_instances.bids = bids
            offset += blen

        sel_p = torch.rand(1)
        
        batch_pseudo_aug = [copy.deepcopy(pb) if pb is not None else None for pb in batch_pseudo_boxes]
        
        if sel_p < self.ss_prob[0]:
            rot = math.pi * (torch.rand(1).item() * (self.rotate_range[1] - self.rotate_range[0]) + self.rotate_range[0])
            for img_metas in batch_img_metas:
                img_metas['ss'] = ('rot', rot)
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            
            batch_inputs_aug, batch_gt_aug, batch_pseudo_aug = self.rotate_crop(
                batch_inputs, rot, [H, W], batch_gt_aug, batch_pseudo_aug, 'reflection')
                
            for gt_instances in batch_gt_aug:
                gt_instances.bids[:, 0] += len(batch_gt_instances)
                gt_instances.bids[:, 2] = 1
                
        elif sel_p < self.ss_prob[0] + self.ss_prob[1]:
            for img_metas in batch_img_metas:
                img_metas['ss'] = ('flp', 0)
            batch_inputs_aug = transforms.functional.vflip(batch_inputs)
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances.bboxes.flip_([H, W], 'vertical')
                gt_instances.bids[:, 0] += len(batch_gt_instances)
                gt_instances.bids[:, 2] = 1
                
            for pb in batch_pseudo_aug:
                if pb is not None:
                    pb.flip_([H, W], 'vertical')
        else:
            sca = (torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
            for img_metas in batch_img_metas:
                img_metas['ss'] = ('sca', sca)
            batch_inputs_aug = transforms.functional.resized_crop(batch_inputs, 0, 0, int(H / sca), int(W / sca), [H, W])
            batch_gt_aug = copy.deepcopy(batch_gt_instances)
            for gt_instances in batch_gt_aug:
                gt_instances.bboxes.rescale_([sca, sca])
                gt_instances.bids[:, 0] += len(batch_gt_instances)
                gt_instances.bids[:, 2] = 1
                
            for pb in batch_pseudo_aug:
                if pb is not None:
                    pb.rescale_([sca, sca])
                
        batch_inputs_all = torch.cat((batch_inputs, batch_inputs_aug))
        self.bbox_head.images = batch_inputs_all
        
        # Edge
        if self.epoch >= self.bbox_head.edge_loss_start_epoch:
            with torch.no_grad():
                mean = self.data_preprocessor.mean
                std = self.data_preprocessor.std
                batch_edges = self.ted_model(batch_inputs_all * std + mean)
                self.bbox_head.edges = batch_edges[3].clamp(0)

        # Copy Paste Logic (Only operates on original GT Instances)
        if self.copy_paste_cache and len(batch_gt_aug) == len(self.copy_paste_cache):
            for i in range(len(batch_gt_aug)):
                gt_instances, patterns = batch_gt_aug[i], self.copy_paste_cache[i]
                bboxes_paste = []
                labels_paste = []
                for p, b, l in patterns:
                    h, w = p.shape[1:3]
                    ox = np.random.randint(0, batch_inputs_aug.shape[-1] - w)
                    oy = np.random.randint(0, batch_inputs_aug.shape[-2] - h)
                    batch_inputs_aug[i, :, oy:oy + h, ox:ox + w] = \
                        batch_inputs_aug[i, :, oy:oy + h, ox:ox + w] * (1 - p[(3,)]) + p[:3] * p[(3,)]
                    bboxes_paste.append(b + np.float32((ox, oy, 0, 0, 0)))
                    labels_paste.append(l)
                bboxes = torch.cat((gt_instances.bboxes.tensor, 
                                    gt_instances.bboxes.tensor.new_tensor(np.float32(bboxes_paste))))
                labels = torch.cat((gt_instances.labels, 
                                    gt_instances.labels.new_tensor(np.int32(labels_paste))))
                bids = torch.cat((gt_instances.bids, 
                                  gt_instances.bids.new_tensor((i, 1, 0, 0)).expand(len(labels_paste), -1)))
                gt_instances = InstanceData()
                gt_instances.bboxes = RotatedBoxes(bboxes)
                gt_instances.labels = labels
                gt_instances.bids = bids
                batch_gt_aug[i] = gt_instances

        batch_inputs_all = torch.cat((batch_inputs, batch_inputs_aug))
        batch_data_samples_all = []
        
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        all_gts = batch_gt_instances + batch_gt_aug
        all_metas = batch_img_metas + batch_img_metas
        all_pseudos = batch_pseudo_boxes + batch_pseudo_aug
        
        for gt_instances, img_metas, pb in zip(all_gts, all_metas, all_pseudos):
            meta_copy = img_metas.copy()
            if pb is not None:
                meta_copy['pseudo_boxes'] = pb.tensor 
            else:
                meta_copy['pseudo_boxes'] = None
                
            data_sample = DetDataSample(metainfo=meta_copy)
            data_sample.gt_instances = gt_instances
            batch_data_samples_all.append(data_sample)
        
        feat = self.extract_feat(batch_inputs_all)
        results_list = self.bbox_head.predict(feat, batch_data_samples_all)
        
        for data_sample, results in zip(batch_data_samples_all, results_list):
            mask = data_sample.gt_instances.bids[:, 1] == 0
            data_sample.gt_instances.bboxes.tensor[mask] = results.bboxes.tensor[mask]
            data_sample.gt_instances.labels[mask] = results.labels[mask]

        losses = self.bbox_head.loss(feat, batch_data_samples_all)

        if self.epoch >= self.copy_paste_start_epoch:
            self.copy_paste_cache = []
            for images, instances in zip(batch_inputs, results_list):
                self.copy_paste_cache.append(get_copy_paste_cache(images, 
                                                                  instances.bboxes.tensor, 
                                                                  instances.labels, 
                                                                  self.bbox_head.square_cls,
                                                                  self.num_copies))
                
        return losses
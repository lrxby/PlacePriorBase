# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss
from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(B, dtype=bool)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tuple): tuple of (xy, sigma).
                xy (torch.Tensor): center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor))


def plot_gaussian_voronoi_watershed(*images):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(len(images) * 4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        if i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
    plt.close()


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres, 
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss, self.vis = gaussian_voronoi_watershed_loss(*pred, 
                                               label,
                                               image, 
                                               pos_thres, 
                                               neg_thres, 
                                               self.down_sample, 
                                               topk=self.topk,
                                               voronoi=voronoi,
                                               alpha=self.alpha,
                                               debug=self.debug)
        return self.loss_weight * loss


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                      (rbbox_concat[:, 2:4] * exy).detach(),
                                      beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)
    
@MODELS.register_module()
class SizeLoss(nn.Module):
    """ ( Top-K ,  - )

    
    1. 
    2. Top-K  topk 
    3.  wh_ratio_balance  ( 1:1 )
    """

    def __init__(self,
                 loss_weight=1.0,
                 ridge_lambda=1e-4,
                 beta=1.0,
                 norm_type='z-score',
                 target_classes=None,
                 topk=1.0,
                 wh_ratio_balance=0.5):
        """
        Args:
            loss_weight (float): .
            ridge_lambda (float):  L2 .
            beta (float): SmoothL1  beta .
            norm_type (str):  ('z-score', 'image-norm'  None).
            target_classes (list[int], optional): .
            topk (float):  (1.0 ).
            wh_ratio_balance (float):  alpha,  1-alpha.
                 0.5 ( 1:1 , ).
        """
        super(SizeLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ridge_lambda = ridge_lambda
        self.beta = beta
        self.norm_type = norm_type
        self.target_classes = target_classes
        self.topk = topk
        self.wh_ratio_balance = wh_ratio_balance
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def _solve_single_image(self, pred_bboxes, scores, labels, img_shape=None):
        # 1. 
        if pred_bboxes.shape[0] == 0:
            return pred_bboxes.new_tensor(0.0)

        x_c = pred_bboxes[:, 0]
        y_c = pred_bboxes[:, 1]
        w = pred_bboxes[:, 2].clamp(min=1e-2)
        h = pred_bboxes[:, 3].clamp(min=1e-2)

        # ----------------------------------------------------------
        #  ()
        # ----------------------------------------------------------
        # []:  torch.log()
        y_w = w
        y_h = h

        #  Y w  h  2N x 1 
        Y = torch.cat([y_w, y_h], dim=0).unsqueeze(1)

        #  v_i 
        # [ 1]: 
        weights = scores.detach().clamp(min=1e-6)
        sqrt_w = torch.sqrt(weights)

        #  2N x 1 
        #  W^{1/2} 
        joint_sqrt_w = torch.cat([sqrt_w, sqrt_w], dim=0).unsqueeze(1)

        # 2. 
        unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
        K = len(unique_labels)
        N = len(pred_bboxes)

        #  2N >= 2K+2 N >= K+3
        if N < K + 3:
            return pred_bboxes.new_tensor(0.0)

        # ----------------------------------------------------------
        #  ()
        # ----------------------------------------------------------
        if self.norm_type == 'z-score':
            x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
            y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
            x_norm = (x_c - x_mean) / x_std
            y_norm = (y_c - y_mean) / y_std
        elif self.norm_type == 'image-norm':
            if img_shape is None:
                x_norm = x_c
                y_norm = y_c
            else:
                H_img, W_img = img_shape[:2]
                x_norm = (x_c - W_img / 2.0) / (W_img / 2.0)
                y_norm = (y_c - H_img / 2.0) / (H_img / 2.0)
        else:
            x_norm = x_c
            y_norm = y_c

        # ----------------------------------------------------------
        #  A
        # ----------------------------------------------------------
        #  2N x (2+2K)  0  A
        A = pred_bboxes.new_zeros((2 * N, 2 + 2 * K))

        #  x, y
        # ( N  N )
        A[:N, 0] = x_norm
        A[N:, 0] = x_norm
        A[:N, 1] = y_norm
        A[N:, 1] = y_norm

        #  scatter_  1  One-Hot 
        # 4.1  (N) ->  2  2+K-1 
        A[:N, 2:2 + K].scatter_(1, labels_inv.unsqueeze(1), 1.0)
        # 4.2  (N) ->  2+K  2+2K-1 
        A[N:, 2 + K:2 + 2 * K].scatter_(1, labels_inv.unsqueeze(1), 1.0)

        # ----------------------------------------------------------

        # ----------------------------------------------------------
        #  Aw  Yw
        #  A  2N x (2+2K)joint_sqrt_w  2N x 1
        A_w = A * joint_sqrt_w
        Y_w = Y * joint_sqrt_w

        #  (A_w^T A_w) (2+2K) x (2+2K)
        M = torch.matmul(A_w.t(), A_w)

        #  λI ( (2+2K)  lambda)
        I_reg = torch.eye(2 + 2 * K, device=pred_bboxes.device) * self.ridge_lambda

        #  M_reg
        M_reg = M + I_reg
        RHS = torch.matmul(A_w.t(), Y_w)

        try:
            #  M_reg * theta = RHS
            #  LU  2  2K 
            theta = torch.linalg.solve(M_reg, RHS)
        except RuntimeError:
            return pred_bboxes.new_tensor(0.0)

        # ----------------------------------------------------------
        # Top-K  Loss
        # ----------------------------------------------------------
        # 1.  Y_hat (2N x 1)
        # [ 2]:  theta ()  Ground Truth
        Y_hat = torch.matmul(A, theta.detach()).squeeze(1)

        # 2.  N  N 
        Y_hat_w = Y_hat[:N]
        Y_hat_h = Y_hat[N:]

        # 3.  SmoothL1 
        loss_w = self.smooth_l1(y_w, Y_hat_w)
        loss_h = self.smooth_l1(y_h, Y_hat_h)

        # 4.  E_total ( alpha)
        #  wh_ratio_balance=0.5 E_total = 0.5*loss_w + 0.5*loss_h
        E_total = self.wh_ratio_balance * loss_w + (1 - self.wh_ratio_balance) * loss_h

        # 5. Top-K  ()
        if self.topk < 1.0:
            K_keep = int(max(1, math.ceil(N * self.topk)))
            if K_keep < N:
                #  E_total K_keep 
                _, topk_indices = torch.topk(E_total, K_keep, largest=False)

                #  mask^(i,c)
                mask = torch.zeros_like(weights, dtype=torch.bool)
                mask[topk_indices] = True

                #  Loss 
                weights = weights * mask.float()

        # 6.  Loss
        loss_weighted = torch.sum(E_total * weights) / (torch.sum(weights) + 1e-6)

        return loss_weighted

    def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
        """
        Forward  Batch 
        """
        # 0. 
        if self.target_classes is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls_id in self.target_classes:
                mask |= (labels == cls_id)
            if mask.sum() == 0:
                return pred_bboxes.new_tensor(0.0)
            pred_bboxes = pred_bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            if batch_idxs is not None:
                batch_idxs = batch_idxs[mask]

        if pred_bboxes.shape[-1] < 4:
            return pred_bboxes.new_tensor(0.0)


        if batch_idxs is None:
            return self.loss_weight * self._solve_single_image(pred_bboxes, scores, labels)

        # Batch 
        unique_batches = torch.unique(batch_idxs)
        total_loss = pred_bboxes.new_tensor(0.0)
        valid_batches = 0.0

        for b_idx in unique_batches:
            mask = (batch_idxs == b_idx)
            b_pred_bboxes = pred_bboxes[mask]
            b_scores = scores[mask]
            b_labels = labels[mask]

            b_img_shape = None
            if img_metas is not None and len(img_metas) > b_idx:
                b_img_shape = img_metas[int(b_idx.item())]['img_shape']

            loss_per_img = self._solve_single_image(b_pred_bboxes, b_scores, b_labels, img_shape=b_img_shape)

            if loss_per_img > 0:
                total_loss += loss_per_img
                valid_batches += 1.0

        if valid_batches > 0:
            return self.loss_weight * (total_loss / valid_batches)
        else:
            return pred_bboxes.new_tensor(0.0)

@MODELS.register_module()
class AngleLoss(nn.Module):
    """AngleLoss  (V5  + TopK )
    
    
    1. 
    2.  (ReLU)
    3.  Duplications (GT ID)
    4.  (Dot Product)
    5. Top-K 
    """

    def __init__(self,
                 loss_weight=1.0,
                 k_radius=2.0,
                 score_alpha=1.0,
                 target_classes=None,
                 reduction='mean',
                 topk=1.0):
        super(AngleLoss, self).__init__()
        self.loss_weight = loss_weight
        self.k_radius = k_radius
        self.score_alpha = score_alpha
        self.target_classes = target_classes
        self.reduction = reduction
        self.topk = topk

    def _forward_single_image(self, bboxes, scores, labels, gt_ids=None):
        N = bboxes.shape[0]
        if N < 2:
            return bboxes.sum() * 0.0, 0.0

        # === Step 1:  ===
        centers = bboxes[:, :2].detach()
        wh = bboxes[:, 2:4].detach()
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = bboxes[:, 4]

        # === Step 2:  (4-Theta) ===
        # vecs 
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)

        # === Step 3:  ===
        dist_sq = torch.cdist(centers, centers, p=2).pow(2)
        sigmas = scales * self.k_radius
        sigma_mat = sigmas.view(N, 1)
        W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2))).detach()

        scores_detached = scores.detach().pow(self.score_alpha)
        W_conf = scores_detached.view(1, N)

        mask_cls = (labels.view(N, 1) == labels.view(1, N)).float()

        # [ 3]:  vecs  .detach()  Loss
        #  W_angle 
        W_angle = torch.mm(vecs.detach(), vecs.detach().t())
        W_angle = torch.relu(W_angle) 

        if gt_ids is not None:
            gt_ids_mat = gt_ids.view(N, 1).expand(N, N)
            mask_duplicate = (gt_ids_mat != gt_ids_mat.t()).float()
        else:
            mask_duplicate = 1.0 - torch.eye(N, device=bboxes.device)

        W = W_geo * W_conf * mask_cls * W_angle * mask_duplicate

        # === Step 4:  ===
        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = W / (W_sum + 1e-6)

        # === Step 5:  (Dot Product) ===
        # [ 4]:  target_dirs
        # ""()
        mean_vecs = torch.mm(W_norm, vecs.detach()) 
        target_dirs = (mean_vecs / (mean_vecs.norm(dim=1, keepdim=True) + 1e-6)).detach()
        
        consistency = (vecs * target_dirs).sum(dim=1)
        chaos_score = 1.0 - consistency

        # === Step 6:  ===
        # 1.  ( & )
        if self.target_classes is not None:
            class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for t_cls in self.target_classes:
                class_mask = class_mask | (labels == t_cls)
            class_mask = class_mask.float()
        else:
            class_mask = torch.ones_like(labels, dtype=torch.float)

        has_neighbor_mask = (W_sum.view(-1) > 1e-6).float()
        
        #  Mask
        final_valid_mask = class_mask * has_neighbor_mask
        
        #  Loss 
        valid_indices = torch.nonzero(final_valid_mask).squeeze()
        
        if valid_indices.numel() == 0:
            return bboxes.sum() * 0.0, 0.0
            
        #  Loss
        active_losses = chaos_score[valid_indices]
        
        # Top-K 
        if self.topk < 1.0:
            num_valid = active_losses.numel()
            #  1 
            num_keep = int(max(1, math.ceil(num_valid * self.topk)))
            
            if num_keep < num_valid:
                #  Loss (largest=False) 
                loss_keep, _ = torch.topk(active_losses, num_keep, largest=False)
                

                return loss_keep.sum(), float(num_keep)

        #  topk 
        final_loss = chaos_score * final_valid_mask
        return final_loss.sum(), final_valid_mask.sum()

    def forward(self,
                pos_bbox_preds,
                pos_scores,
                pos_labels,
                batch_idxs,
                pos_gt_ids=None, 
                **kwargs):
        total_loss = 0.0
        total_valid_samples = 0.0

        unique_batch_ids = torch.unique(batch_idxs)

        for b_id in unique_batch_ids:
            mask = (batch_idxs == b_id)
            if mask.sum() == 0: continue

            img_bboxes = pos_bbox_preds[mask]
            img_scores = pos_scores[mask]
            img_labels = pos_labels[mask]
            
            img_gt_ids = None
            if pos_gt_ids is not None:
                img_gt_ids = pos_gt_ids[mask]

            loss_sum, valid_count = self._forward_single_image(
                img_bboxes, img_scores, img_labels, img_gt_ids
            )

            total_loss += loss_sum
            total_valid_samples += valid_count

        if self.reduction == 'mean':
            if total_valid_samples > 0:
                return self.loss_weight * total_loss / total_valid_samples
            else:
                return pos_bbox_preds.sum() * 0.0
        else:
            return self.loss_weight * total_loss

@MODELS.register_module()
class OurWaterLoss(nn.Module):
    """
    
    
     Loss  (Pred RBoxes)  (Pseudo RBoxes) 
     Wasserstein  (Gaussian Wasserstein Distance)
    
    
    1.  (W/H)  (Angle) 
    2.  (X/Y)  Point  Heatmap Loss 
    3.  GWD 
    
    Args:
        loss_weight (float):  1.0
    """
    def __init__(self, loss_weight=1.0):
        super(OurWaterLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self, pred_rboxes, pseudo_rboxes, weight=None, avg_factor=None, **kwargs):
        """
        

        Args:
            pred_rboxes (Tensor): [N, 5]  (x, y, w, h, theta)
            pseudo_rboxes (Tensor): [N, 5]  (x, y, w, h, theta) pkl 
            weight (Tensor, optional): [N]  Assigner 
            avg_factor (float, optional):  Loss 
        """
        # 0.  Batch 
        if pred_rboxes.numel() == 0:
            return pred_rboxes.sum() * 0
            
        # 1.  Sigma
        # (w, h, theta) -> Sigma (2x2)

        sigma_p = self.rbox2sigma_batch(pred_rboxes)
        sigma_t = self.rbox2sigma_batch(pseudo_rboxes)
        
        # 2.  GWD Sigma Loss
        #  gwd_sigma_loss  reduction, weight  avg_factor
        # gwd_sigma_loss  MMDetection  @weighted_loss 
        #  Loss  reduction 
        #   a) 
        #   b)  Loss 
        #  element-wise ()  Loss 
        # fun='log1p' log(1+x) 
        loss_vector = gwd_sigma_loss(sigma_p, sigma_t, fun='log1p')
        
        # 3.  (Weighted Mean)
        if weight is not None:
            loss_vector = loss_vector * weight
            
        if avg_factor is not None:
            return self.loss_weight * loss_vector.sum() / avg_factor
        else:
            return self.loss_weight * loss_vector.mean()

    def rbox2sigma_batch(self, rboxes):
        """
        
        
        : Sigma = R * Lambda * R^T
         R Lambda  ()

        Args:
            rboxes (Tensor): [N, 5] 

        Returns:
            sigma (Tensor): [N, 2, 2] 
        """
        #  w, h, angle
        #  clamp(min=1e-4) 
        w = rboxes[:, 2].clamp(min=1e-4)
        h = rboxes[:, 3].clamp(min=1e-4)
        angle = rboxes[:, 4]
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        #  R
        # R = [[cos, -sin], 
        #      [sin,  cos]]
        # Shape: (N, 2, 2)
        row1 = torch.stack([cos, -sin], dim=-1)
        row2 = torch.stack([sin, cos], dim=-1)
        R = torch.stack([row1, row2], dim=-2)
        
        #  Lambda = diag((w/2)^2, (h/2)^2)
        # /
        # Shape: (N, 2, 2)
        var_x = (w / 2).pow(2)
        var_y = (h / 2).pow(2)
        zeros = torch.zeros_like(var_x)
        
        lambda_mat = torch.stack([
            torch.stack([var_x, zeros], dim=-1),
            torch.stack([zeros, var_y], dim=-1)
        ], dim=-2)
        
        #  Sigma
        # Sigma = R @ Lambda @ R_T
        # bmm:  (Batch Matrix Multiplication)
        # (N, 2, 2) x (N, 2, 2) -> (N, 2, 2)
        sigma = torch.bmm(R, torch.bmm(lambda_mat, R.transpose(1, 2)))
        
        return sigma
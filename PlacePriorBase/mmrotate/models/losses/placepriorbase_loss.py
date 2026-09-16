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








import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS

@MODELS.register_module()
class SizePriorLoss(nn.Module):
    """Perspective-aware size consistency loss (with Top-K relaxation).

    Joint ridge regression on width and height with a shared perspective slope
    and per-class intercepts. Features:
    1. Dual-target joint ridge regression: fits width and height together.
    2. Top-K mechanism: gradients only on the best-fitted topk samples.
    3. Width/height balance via wh_ratio_balance (default 1:1).
    """

    def __init__(self,
                 loss_weight=1.0,
                 ridge_lambda=1e-4,
                 beta=1.0,
                 norm_type='z-score',
                 target_classes=None,
                 topk=1.0,
                 wh_ratio_balance=0.5,
                 reg_space='log',
                 loss_type='smoothl1',
                 scale_factor=1.0):
        """
        Args:
            loss_weight (float): total loss weight.
            ridge_lambda (float): ridge (L2) regularization coefficient.
            beta (float): beta of the SmoothL1 loss.
            norm_type (str): coordinate normalization ('z-score', 'image-norm' or None).
            target_classes (list[int], optional): only compute the loss for these classes.
            topk (float): keep the fraction of samples with the smallest residuals (1.0 = all).
            wh_ratio_balance (float): weight alpha for width; height gets 1 - alpha.
                Default 0.5 (equal 1:1, most robust).
        """
        super(SizePriorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ridge_lambda = ridge_lambda
        self.beta = beta
        self.norm_type = norm_type
        self.target_classes = target_classes
        self.topk = topk
        self.wh_ratio_balance = wh_ratio_balance
        self.reg_space = reg_space
        self.loss_type = loss_type
        self.scale_factor = scale_factor
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def _solve_single_image(self, pred_bboxes, scores, labels, img_shape=None):
        # 1. prepare data
        if pred_bboxes.shape[0] == 0:
            return pred_bboxes.new_tensor(0.0)

        x_c = pred_bboxes[:, 0]
        y_c = pred_bboxes[:, 1]
        w = pred_bboxes[:, 2].clamp(min=1e-2)
        h = pred_bboxes[:, 3].clamp(min=1e-2)

        # ----------------------------------------------------------
        # Part 2: log-scale mapping and weight assignment (dual-target)
        # ----------------------------------------------------------
        # per-dimension mapping: linear regression on sizes; log space by default
        if self.reg_space == 'linear':
            y_w = w
            y_h = h
        else:
            y_w = torch.log(w)
            y_h = torch.log(h)

        # target vector Y: concatenate log w and log h into a 2N x 1 vector
        Y = torch.cat([y_w, y_h], dim=0).unsqueeze(1)

        # per-dimension weight vector v_i
        # detach classification scores so regression errors do not suppress confidence
        weights = scores.detach().clamp(min=1e-6)
        sqrt_w = torch.sqrt(weights)

        # concatenate the weights into a 2N x 1 vector
        # broadcasting implements the block-diagonal W^{1/2} multiplication
        joint_sqrt_w = torch.cat([sqrt_w, sqrt_w], dim=0).unsqueeze(1)

        # 2. check degrees of freedom
        unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
        K = len(unique_labels)
        N = len(pred_bboxes)

        # joint regression needs 2N >= 2K+2; keep the redundant constraint N >= K+3
        if N < K + 3:
            return pred_bboxes.new_tensor(0.0)

        # ----------------------------------------------------------
        # Part 3: spatial coordinate normalization
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
        # Part 4: build the class-decoupled joint design matrix A
        # ----------------------------------------------------------
        # build the 2N x (2+2K) zero joint design matrix A
        A = pred_bboxes.new_zeros((2 * N, 2 + 2 * K))

        # fill the shared perspective features x, y
        # (both halves reuse the same coordinates, so the perspective slope is shared)
        A[:N, 0] = x_norm
        A[N:, 0] = x_norm
        A[:N, 1] = y_norm
        A[N:, 1] = y_norm

        # use scatter_ to place ones for the one-hot class intercepts
        # 4.1 top half (first N rows) -> width intercepts at columns 2..2+K-1
        A[:N, 2:2 + K].scatter_(1, labels_inv.unsqueeze(1), 1.0)
        # 4.2 bottom half (last N rows) -> height intercepts at columns 2+K..2+2K-1
        A[N:, 2 + K:2 + 2 * K].scatter_(1, labels_inv.unsqueeze(1), 1.0)

        # ----------------------------------------------------------
        # Part 5: weighted ridge regression
        # ----------------------------------------------------------
        # build the weighted design matrix Aw and target vector Yw
        # A is 2N x (2+2K), joint_sqrt_w is 2N x 1; broadcasting applies
        A_w = A * joint_sqrt_w
        Y_w = Y * joint_sqrt_w

        # covariance matrix (A_w^T A_w) -> (2+2K) x (2+2K)
        M = torch.matmul(A_w.t(), A_w)

        # regularization term lambda * I of size (2+2K)
        I_reg = torch.eye(2 + 2 * K, device=pred_bboxes.device) * self.ridge_lambda

        # combined coefficient matrix M_reg is full-rank and invertible
        M_reg = M + I_reg
        RHS = torch.matmul(A_w.t(), Y_w)

        try:
            # solve the normal equations M_reg * theta = RHS
            # LU decomposition solves for 2 shared slopes and 2K intercepts at once
            theta = torch.linalg.solve(M_reg, RHS)
        except RuntimeError:
            return pred_bboxes.new_tensor(0.0)

        # ----------------------------------------------------------
        # Part 6: residuals, Top-K masking, and final loss
        # ----------------------------------------------------------
        # 1. predicted values Y_hat (2N x 1)
        # detach the fitted regression plane and treat it as fixed ground truth
        Y_hat = torch.matmul(A, theta.detach()).squeeze(1)

        # 2. split: first N are width predictions, last N are height predictions
        Y_hat_w = Y_hat[:N]
        Y_hat_h = Y_hat[N:]

        # 3. width and height residuals (L2; SmoothL1 by default)
        if self.loss_type == 'l2':
            loss_w = F.mse_loss(y_w, Y_hat_w, reduction='none')
            loss_h = F.mse_loss(y_h, Y_hat_h, reduction='none')
        else:
            loss_w = self.smooth_l1(y_w, Y_hat_w)
            loss_h = self.smooth_l1(y_h, Y_hat_h)

        # 4. combined per-sample residual E_total with balance alpha
        # wh_ratio_balance=0.5 -> E_total = 0.5*loss_w + 0.5*loss_h
        # the 0.5 factors cancel in the weighted mean
        E_total = self.wh_ratio_balance * loss_w + (1 - self.wh_ratio_balance) * loss_h

        # 5. Top-K masking (blocks gradients on both width and height)
        if self.topk < 1.0:
            K_keep = int(max(1, math.ceil(N * self.topk)))
            if K_keep < N:
                # indices of the K_keep samples with the smallest E_total
                _, topk_indices = torch.topk(E_total, K_keep, largest=False)

                # survival mask mask^(i,c)
                mask = torch.zeros_like(weights, dtype=torch.bool)
                mask[topk_indices] = True

                # zero the weights of outlier samples
                weights = weights * mask.float()

        # 6. final weighted mean loss
        loss_weighted = torch.sum(E_total * weights) / (torch.sum(weights) + 1e-6)

        return loss_weighted

    def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
        """
        Forward: dispatch per batch to single images and filter target classes.
        """
        # 0. filter target classes
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

        # single image: solve directly
        if batch_idxs is None:
            return self.loss_weight * self.scale_factor * self._solve_single_image(pred_bboxes, scores, labels)

        # batch processing: solve per image
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
            return self.loss_weight * self.scale_factor * (total_loss / valid_batches)
        else:
            return pred_bboxes.new_tensor(0.0)


@MODELS.register_module()
class AnglePriorLoss(nn.Module):
    """Angle consistency loss (with Top-K relaxation).
    
    Features:
    1. Computed per image.
    2. Angle grouping (ReLU).
    3. Duplications avoided via GT ID.
    4. Direction consistency via dot product.
    5. Top-K relaxation: drop outliers with poor local consistency.
    """

    def __init__(self,
                 loss_weight=1.0,
                 k_radius=2.0,
                 score_alpha=1.0,
                 target_classes=None,
                 reduction='mean',
                 topk=1.0,
                 warmup_epochs=0,
                 neighbor_metric='euclidean_sq',
                 eps=1e-6,
                 scale_factor=1.0):
        super(AnglePriorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.k_radius = k_radius
        self.score_alpha = score_alpha
        self.target_classes = target_classes
        self.reduction = reduction
        self.topk = topk
        self.warmup_epochs = warmup_epochs
        self.neighbor_metric = neighbor_metric
        self.eps = eps
        self.scale_factor = scale_factor
        self.current_epoch = 0

    def _forward_single_image(self, bboxes, scores, labels, gt_ids=None):
        N = bboxes.shape[0]
        if N < 2:
            return bboxes.sum() * 0.0, 0.0

        # === Step 1: geometric decoupling ===
        centers = bboxes[:, :2].detach()
        wh = bboxes[:, 2:4].detach()
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = bboxes[:, 4]

        # === Step 2: vectorization (4-theta) ===
        # vecs keep gradients until the final penalty
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)

        # === Step 3: affinity matrix ===
        if self.neighbor_metric == 'manhattan':
            # W(i,j) = exp(-(|dx|+|dy|) / (2 * w_i * h_i))
            dist_manh = torch.cdist(centers, centers, p=1)
            wh_i = (wh[:, 0] * wh[:, 1]).clamp(min=1.0).view(N, 1)
            W_geo = torch.exp(-dist_manh / (2 * wh_i)).detach()
        else:
            dist_sq = torch.cdist(centers, centers, p=2).pow(2)
            sigmas = scales * self.k_radius
            sigma_mat = sigmas.view(N, 1)
            W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2))).detach()

        scores_detached = scores.detach().pow(self.score_alpha)
        W_conf = scores_detached.view(1, N)

        mask_cls = (labels.view(N, 1) == labels.view(1, N)).float()

        # detach vecs for the affinity matrix so the network cannot game W_angle
        # by predicting mutually orthogonal targets.
        W_angle = torch.mm(vecs.detach(), vecs.detach().t())
        W_angle = torch.relu(W_angle) 

        if gt_ids is not None:
            gt_ids_mat = gt_ids.view(N, 1).expand(N, N)
            mask_duplicate = (gt_ids_mat != gt_ids_mat.t()).float()
        else:
            mask_duplicate = 1.0 - torch.eye(N, device=bboxes.device)

        W = W_geo * W_conf * mask_cls * W_angle * mask_duplicate

        # === Step 4: normalization ===
        W_sum = W.sum(dim=1, keepdim=True)
        W_norm = W / (W_sum + 1e-6)

        # === Step 5: energy (dot product) ===
        # compute target_dirs outside the graph so objects move toward the
        # local consensus instead of twisting it (avoid mode collapse).
        mean_vecs = torch.mm(W_norm, vecs.detach()) 
        target_dirs = (mean_vecs / (mean_vecs.norm(dim=1, keepdim=True) + self.eps)).detach()
        
        consistency = (vecs * target_dirs).sum(dim=1)
        chaos_score = 1.0 - consistency

        # === Step 6: final filtering and relaxation ===
        # 1. basic filtering (class & outliers)
        if self.target_classes is not None:
            class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for t_cls in self.target_classes:
                class_mask = class_mask | (labels == t_cls)
            class_mask = class_mask.float()
        else:
            class_mask = torch.ones_like(labels, dtype=torch.float)

        has_neighbor_mask = (W_sum.view(-1) > 1e-6).float()
        
        # mask of valid samples
        final_valid_mask = class_mask * has_neighbor_mask
        
        # loss of all valid samples
        valid_indices = torch.nonzero(final_valid_mask).squeeze()
        
        if valid_indices.numel() == 0:
            return bboxes.sum() * 0.0, 0.0
            
        # valid losses
        active_losses = chaos_score[valid_indices]
        
        # Top-K relaxation
        if self.topk < 1.0:
            num_valid = active_losses.numel()
            # keep at least 1
            num_keep = int(max(1, math.ceil(num_valid * self.topk)))
            
            if num_keep < num_valid:
                # keep the smallest losses (largest=False)
                loss_keep, _ = torch.topk(active_losses, num_keep, largest=False)
                
                # truncated sum and sample count
                return loss_keep.sum(), float(num_keep)

        # return directly if topk is disabled or too few samples
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

        # Warmup: linearly increase loss weight over first warmup_epochs
        # If warmup_epochs == 0, warmup is disabled (full weight from start)
        if self.warmup_epochs > 0:
            warmup_w = min(1.0, (self.current_epoch + 1) / self.warmup_epochs)
        else:
            warmup_w = 1.0
        effective_weight = warmup_w * self.loss_weight * self.scale_factor

        if self.reduction == 'mean':
            if total_valid_samples > 0:
                return effective_weight * total_loss / total_valid_samples
            else:
                return pos_bbox_preds.sum() * 0.0
        else:
            return effective_weight * total_loss

@MODELS.register_module()
class VWRBoxLoss(nn.Module):
    """
    Offline pseudo-label supervision loss.
    
    This loss measures the Gaussian Wasserstein Distance (GWD) between
    the predicted rotated boxes and the offline pseudo rotated boxes.
    
    Main functions:
    1. Joint geometric supervision on shape (W/H) and orientation (Angle).
    2. Ignores the center (X/Y) difference (usually handled by the point branch).
    3. GWD naturally handles ambiguous width/height definitions and angle periodicity.
    
    Args:
        loss_weight (float): weight of the loss. Default 1.0.
    """
    def __init__(self, loss_weight=1.0, topk=1.0, scale_factor=1.0):
        super(VWRBoxLoss, self).__init__()
        self.loss_weight = loss_weight
        self.topk = topk
        self.scale_factor = scale_factor
        
    def forward(self, pred_rboxes, pseudo_rboxes, weight=None, avg_factor=None, **kwargs):
        """
        Forward pass.

        Args:
            pred_rboxes (Tensor): [N, 5] tensor (x, y, w, h, theta) from the network.
            pseudo_rboxes (Tensor): [N, 5] tensor (x, y, w, h, theta) from the offline pkl.
            weight (Tensor, optional): [N] per-sample weights (usually from the assigner).
            avg_factor (float, optional): average factor for loss normalization.
        """
        # 0. safety check for an empty batch
        if pred_rboxes.numel() == 0:
            return pred_rboxes.sum() * 0
            
        # 1. convert rotated boxes to Gaussian covariance Sigma
        # (w, h, theta) -> Sigma (2x2)
        # map a rotated box to a 2D Gaussian for distribution-based distance
        sigma_p = self.rbox2sigma_batch(pred_rboxes)
        sigma_t = self.rbox2sigma_batch(pseudo_rboxes)
        
        # 2. GWD sigma loss
        # Implementation note: do not pass reduction/weight/avg_factor to
        # gwd_sigma_loss here; it is wrapped by mmdet's @weighted_loss decorator.
        # Passing reduction again may cause a decorator argument conflict or
        # double weighting. gwd_sigma_loss defaults to reduction='mean' and
        # returns a scalar; pass reduction='none' explicitly to obtain the
        # per-sample vector before re-weighting/filtering.
        # fun='log1p': log(1+x) on distances for numerical stability.
        loss_vector = gwd_sigma_loss(sigma_p, sigma_t, fun='log1p',
                                       reduction='none')
        
        # 3. top-k relaxation: omit the top-10% loss values
        if self.topk < 1.0:
            num_valid = loss_vector.numel()
            num_keep = int(max(1, math.ceil(num_valid * self.topk)))
            if num_keep < num_valid:
                loss_keep, _ = torch.topk(loss_vector, num_keep, largest=False)
                loss_vector = loss_keep

        # 4. manual weighted-mean normalization
        if weight is not None:
            loss_vector = loss_vector * weight

        if avg_factor is not None:
            return self.loss_weight * self.scale_factor * loss_vector.sum() / avg_factor
        else:
            return self.loss_weight * self.scale_factor * loss_vector.mean()

    def rbox2sigma_batch(self, rboxes):
        """
        Convert rotated boxes to Gaussian covariance matrices.
        
        Formula: Sigma = R * Lambda * R^T
        where R is the rotation matrix and Lambda is the diagonal eigenvalue matrix.

        Args:
            rboxes (Tensor): [N, 5] input boxes.

        Returns:
            sigma (Tensor): [N, 2, 2] covariance matrices.
        """
        # extract w, h, angle
        # clamp to avoid numerical instability for tiny boxes
        w = rboxes[:, 2].clamp(min=1e-4)
        h = rboxes[:, 3].clamp(min=1e-4)
        angle = rboxes[:, 4]
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        # rotation matrix R
        # R = [[cos, -sin], 
        #      [sin,  cos]]
        # Shape: (N, 2, 2)
        row1 = torch.stack([cos, -sin], dim=-1)
        row2 = torch.stack([sin, cos], dim=-1)
        R = torch.stack([row1, row2], dim=-2)
        
        # eigenvalue matrix Lambda = diag((w/2)^2, (h/2)^2)
        # variance scales with the squared semi-axes
        # Shape: (N, 2, 2)
        var_x = (w / 2).pow(2)
        var_y = (h / 2).pow(2)
        zeros = torch.zeros_like(var_x)
        
        lambda_mat = torch.stack([
            torch.stack([var_x, zeros], dim=-1),
            torch.stack([zeros, var_y], dim=-1)
        ], dim=-2)
        
        # covariance matrix Sigma
        # Sigma = R @ Lambda @ R_T
        # bmm: batch matrix multiplication
        # (N, 2, 2) x (N, 2, 2) -> (N, 2, 2)
        sigma = torch.bmm(R, torch.bmm(lambda_mat, R.transpose(1, 2)))
        
        return sigma
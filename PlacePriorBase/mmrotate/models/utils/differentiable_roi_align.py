# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DifferentiableRoIAlignRotated(nn.Module):
#     """
#     Pure PyTorch differentiable rotated RoI Align.
#     Supports gradient flow to the ROI coordinates (x, y, w, h, theta).
#     Fully aligned with mmcv.ops.RoIAlignRotated (aligned=True).
#     """
#     def __init__(self, output_size, spatial_scale, sampling_ratio=2, clockwise=True):
#         """
#         Args:
#             output_size (tuple): (h, w)
#             spatial_scale (float): feature map scale (e.g., 1/8)
#             sampling_ratio (int): sampling points per bin (0 means auto; usually 2).
#             clockwise (bool): rotate clockwise; MMRotate defaults to True.
#         """
#         super(DifferentiableRoIAlignRotated, self).__init__()
#         # handle an int output_size
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             self.output_size = output_size
            
#         self.spatial_scale = spatial_scale
#         self.sampling_ratio = sampling_ratio if sampling_ratio > 0 else 1
#         self.clockwise = clockwise

#     def forward(self, features, rois):
#         """
#         Args:
#             features: (N, C, H, W)
#             rois: (K, 6) [batch_ind, x, y, w, h, theta]
#         """
#         num_rois = rois.shape[0]
#         batch_size, channels, height, width = features.shape
#         out_h, out_w = self.output_size
        
#         # oversampled grid
#         grid_h = out_h * self.sampling_ratio
#         grid_w = out_w * self.sampling_ratio

#         # 1. parse ROIs
#         batch_inds = rois[:, 0].long()
        
#         # ================= [key fix] =================
#         # wrong: rois_feat = rois[:, 1:] * self.spatial_scale (theta gets scaled)
#         # correct: handle coordinates/sizes and the angle separately
        
#         # coordinates and sizes (x, y, w, h) map to the feature scale -> multiply by spatial_scale
#         cx = rois[:, 1] * self.spatial_scale
#         cy = rois[:, 2] * self.spatial_scale
#         w  = rois[:, 3] * self.spatial_scale
#         h  = rois[:, 4] * self.spatial_scale
        
#         # the angle (theta) is a geometric property and must not be scaled -> keep it unchanged
#         theta = rois[:, 5] 
#         # ================= [key fix end] ===================

#         # 2. build the sampling grid (affine transform)
#         # bin centers in [-0.5, 0.5] * w/h
#         # simulate uniform sampling inside the box
#         _y = (torch.arange(grid_h, dtype=rois.dtype, device=rois.device) + 0.5) / grid_h - 0.5
#         _x = (torch.arange(grid_w, dtype=rois.dtype, device=rois.device) + 0.5) / grid_w - 0.5
        
#         _y, _x = torch.meshgrid(_y, _x, indexing='ij')
#         # (K, grid_h * grid_w)
#         grid_x = _x.reshape(-1).unsqueeze(0).expand(num_rois, -1)
#         grid_y = _y.reshape(-1).unsqueeze(0).expand(num_rois, -1)

#         # rotation matrix
#         if self.clockwise:
#             cos_t = torch.cos(theta)
#             sin_t = torch.sin(theta)
#         else:
#             cos_t = torch.cos(-theta)
#             sin_t = torch.sin(-theta)

#         # affine transform
#         # scale: grid (-0.5..0.5) * w
#         gx = grid_x * w.unsqueeze(1) 
#         gy = grid_y * h.unsqueeze(1)

#         # rotate + translate to absolute feature coordinates (real_x, real_y)
#         # x' = x*cos - y*sin + cx
#         x_sample = gx * cos_t.unsqueeze(1) - gy * sin_t.unsqueeze(1) + cx.unsqueeze(1)
#         y_sample = gx * sin_t.unsqueeze(1) + gy * cos_t.unsqueeze(1) + cy.unsqueeze(1)

#         # 3. normalize to [-1, 1] for grid_sample
#         # matches mmcv aligned=True (align_corners=False)
#         # formula: coordinate / width * 2 - 1
#         x_grid = (x_sample / width) * 2.0 - 1.0
#         y_grid = (y_sample / height) * 2.0 - 1.0
        
#         # (K, grid_h, grid_w, 2)
#         grid = torch.stack([x_grid, y_grid], dim=2).view(num_rois, grid_h, grid_w, 2)

#         # 4. sampling (batch loop)
#         # output: (K, C, grid_h, grid_w)
#         output_raw = torch.zeros(num_rois, channels, grid_h, grid_w, 
#                                  dtype=features.dtype, device=features.device)
        
#         # sample per batch index
#         # grid_sample requires matching batch dims for input and grid
#         for b_idx in range(batch_size):
#             mask = (batch_inds == b_idx)
#             if not mask.any(): continue
            
#             batch_grid = grid[mask] 
#             # current image features (1, C, H, W)
#             batch_feat = features[b_idx:b_idx+1]
            
#             # expand input to match the number of ROIs
#             batch_feat_expanded = batch_feat.expand(mask.sum(), -1, -1, -1)
            
#             # sample (grid_sample is differentiable w.r.t. the grid)
#             batch_out = F.grid_sample(batch_feat_expanded, batch_grid, 
#                                       mode='bilinear', padding_mode='zeros', align_corners=False)
            
#             output_raw[mask] = batch_out

#         # 5. avg-pool the oversampled features to the target size
#         if self.sampling_ratio > 1:
#             output = F.avg_pool2d(output_raw, kernel_size=self.sampling_ratio, stride=self.sampling_ratio)
#         else:
#             output = output_raw

#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableRoIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio=2, clockwise=True, chunk_size=1000):
        super(DifferentiableRoIAlignRotated, self).__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio if sampling_ratio > 0 else 1
        self.clockwise = clockwise
        self.chunk_size = chunk_size  # internal chunk size

    def forward(self, features, rois):
        # too few ROIs: use the original path
        if rois.shape[0] <= self.chunk_size:
            return self._do_roi_align(features, rois)
        
        # too many ROIs: process in chunks
        results = []
        for i in range(0, rois.shape[0], self.chunk_size):
            chunk_rois = rois[i : i + self.chunk_size]
            # per-chunk computation saves memory
            chunk_out = self._do_roi_align(features, chunk_rois)
            results.append(chunk_out)
        
        return torch.cat(results, dim=0)

    def _do_roi_align(self, features, rois):
        """Original core computation."""
        num_rois = rois.shape[0]
        batch_size, channels, height, width = features.shape
        out_h, out_w = self.output_size
        grid_h, grid_w = out_h * self.sampling_ratio, out_w * self.sampling_ratio

        # 1. parse ROIs
        batch_inds = rois[:, 0].long()
        cx = rois[:, 1] * self.spatial_scale
        cy = rois[:, 2] * self.spatial_scale
        w  = rois[:, 3] * self.spatial_scale
        h  = rois[:, 4] * self.spatial_scale
        theta = rois[:, 5]

        # 2. build the sampling grid
        _y = (torch.arange(grid_h, dtype=rois.dtype, device=rois.device) + 0.5) / grid_h - 0.5
        _x = (torch.arange(grid_w, dtype=rois.dtype, device=rois.device) + 0.5) / grid_w - 0.5
        _y, _x = torch.meshgrid(_y, _x, indexing='ij')
        grid_x = _x.reshape(-1).unsqueeze(0).expand(num_rois, -1)
        grid_y = _y.reshape(-1).unsqueeze(0).expand(num_rois, -1)

        cos_t = torch.cos(theta) if self.clockwise else torch.cos(-theta)
        sin_t = torch.sin(theta) if self.clockwise else torch.sin(-theta)

        gx, gy = grid_x * w.unsqueeze(1), grid_y * h.unsqueeze(1)
        x_sample = gx * cos_t.unsqueeze(1) - gy * sin_t.unsqueeze(1) + cx.unsqueeze(1)
        y_sample = gx * sin_t.unsqueeze(1) + gy * cos_t.unsqueeze(1) + cy.unsqueeze(1)

        # 3. normalize and build the grid
        x_grid, y_grid = (x_sample / width) * 2.0 - 1.0, (y_sample / height) * 2.0 - 1.0
        grid = torch.stack([x_grid, y_grid], dim=2).view(num_rois, grid_h, grid_w, 2)

        # 4. sample
        output_raw = torch.zeros(num_rois, channels, grid_h, grid_w, dtype=features.dtype, device=features.device)
        unique_batches = batch_inds.unique()
        for b_idx in unique_batches:
            mask = (batch_inds == b_idx)
            batch_feat = features[b_idx : b_idx + 1].expand(mask.sum(), -1, -1, -1)
            output_raw[mask] = F.grid_sample(batch_feat, grid[mask], mode='bilinear', padding_mode='zeros', align_corners=False)

        # 5. pool
        return F.avg_pool2d(output_raw, self.sampling_ratio) if self.sampling_ratio > 1 else output_raw
# GT center anchoring
import torch
import numpy as np
import cv2
import os
import tqdm
import pickle
import sys
import math
import torch.nn.functional as F

# ==============================================================================
# 1. configuration (DroneVehicle)
# ==============================================================================
DATA_ROOT = 'data/split_ss_dronevehicle/trainval'
SAVE_PATH = 'data/split_ss_dronevehicle/trainval/dronevehicle-Rect.pkl'
DEVICE = 'cuda:0'

# Method: Row6_Rect (minAreaRect + GT-point-center anchoring)
METHOD = 'Row6_Rect'

# best parameters for DroneVehicle
CFG = dict(sigma=1024, ds=2, pos=0.97, neg=0.1)

# class definitions (for index mapping)
CLASSES = ('car', 'bus', 'truck', 'van', 'freight_car')

# ==============================================================================
# 2. core functions
# ==============================================================================

def rbox2sigma(w, h, angle, device):
    w = torch.tensor(w, device=device, dtype=torch.float32)
    h = torch.tensor(h, device=device, dtype=torch.float32)
    angle = torch.tensor(angle, device=device, dtype=torch.float32)
    cos_a = torch.cos(angle); sin_a = torch.sin(angle)
    R = torch.stack([torch.stack([cos_a, -sin_a], -1), torch.stack([sin_a, cos_a], -1)], -2)
    Lambda = torch.stack([torch.stack([(w/2)**2, torch.zeros_like(w)], -1), 
                          torch.stack([torch.zeros_like(w), (h/2)**2], -1)], -2)
    return R @ Lambda @ R.transpose(-1, -2)

def get_watershed_wh_original_loss(mask_pts, center_pt, angle_rad):
    if len(mask_pts) < 2: return 16.0, 16.0
    pts_centered = mask_pts - center_pt
    cos_inv = np.cos(-angle_rad); sin_inv = np.sin(-angle_rad)
    R_inv = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])
    pts_rot = np.dot(pts_centered, R_inv.T)
    w = max(2 * np.max(np.abs(pts_rot[:, 0])), 4.0)
    h = max(2 * np.max(np.abs(pts_rot[:, 1])), 4.0)
    return w, h

def gaussian_2d(xy, mu, sigma):
    dxy = (xy - mu).unsqueeze(-1)
    inv_sigma = torch.linalg.solve(sigma + torch.eye(2, device=sigma.device)*1e-5, dxy)
    return torch.exp(-0.5 * dxy.permute(0, 1, 3, 2).matmul(inv_sigma)).squeeze()

def run_watershed_generation(sigmas, gt_points, img_shape, pos_thres, neg_thres, down_sample, device):
    H, W = img_shape[:2]; D = down_sample; J = len(gt_points)
    h_map, w_map = H // D, W // D
    if J == 0: return np.zeros((H, W), dtype=np.int32)
    
    ax_x = torch.linspace(0, w_map-1, w_map, device=device)
    ax_y = torch.linspace(0, h_map-1, h_map, device=device)
    grid_x, grid_y = torch.meshgrid(ax_x, ax_y, indexing='xy')
    xy = torch.stack([grid_x, grid_y], -1)
    mm = (torch.tensor(gt_points, device=device).float() / D).round()
    sg = sigmas / (D**2)
    
    vor = torch.zeros((J, h_map, w_map), device=device)
    for j in range(J): vor[j] = gaussian_2d(xy, mm[j], sg[j])
        
    val, vor_idx = torch.max(vor, 0)
    if D > 1:
        vor_idx = F.interpolate(vor_idx[None, None].float(), size=(H, W), mode='nearest')[0, 0].long()
        val = F.interpolate(val[None, None], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
    kernel = torch.ones((1, 1, 3, 3), device=device); kernel[0,0,1,1] = -8
    ridges = torch.conv2d(vor_idx.float()[None,None], kernel, padding=1)[0,0] != 0
    
    vor_idx += 1
    vor_idx[val < pos_thres] = 0
    vor_idx[val < neg_thres] = J + 1
    vor_idx[ridges] = J + 1
    return vor_idx.cpu().numpy().astype(np.int32)

def analyze_geometry_specific(mask_stage1, gt_points, default_sigma, device, method):
    """DroneVehicle geometry analysis core."""
    J = len(gt_points)
    sigmas_list, angles_list = [], []
    for j in range(J):
        pts_y, pts_x = np.where(mask_stage1 == j + 1)
        if len(pts_x) < 5:
            sigmas_list.append(torch.eye(2, device=device) * default_sigma)
            angles_list.append(0.0)
            continue
            
        # ensure correct pixel extraction
        pts_y, pts_x = np.where(mask_stage1 == j + 1)
        mask_pts = np.stack([pts_x, pts_y], axis=1).astype(np.float32)
        gt_center = gt_points[j]
        
        if method == 'Row6_Rect':
            rect = cv2.minAreaRect(mask_pts)
            _, _, angle_deg_rect = rect
            # physically align: correct the OpenCV angle
            if angle_deg_rect > 45: angle_deg_rect -= 90
            theta = np.radians(angle_deg_rect)
        else:
            theta = 0.0
        
        w_raw, h_raw = get_watershed_wh_original_loss(mask_pts, gt_center, theta)
        area_raw = (w_raw / 2) * (h_raw / 2)
        scale = np.sqrt(default_sigma / (area_raw + 1e-6))
        
        sigmas_list.append(rbox2sigma(w_raw*scale, h_raw*scale, theta, device))
        angles_list.append(theta)
        
    return torch.stack(sigmas_list), np.array(angles_list)

# ==============================================================================
# 3. main flow
# ==============================================================================

def generate_dronevehicle():
    print(f"Start Generating DroneVehicle Pseudo Labels...")
    print(f"Selected Method: {METHOD} | Config: {CFG}")
    
    img_dir = os.path.join(DATA_ROOT, 'images')
    lbl_dir = os.path.join(DATA_ROOT, 'annfiles')
        
    files = sorted([os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith('.txt')])
    pseudo_data_dict = {}
    total_objs = 0
    
    for file_id in tqdm.tqdm(files):
        img_path = os.path.join(img_dir, f"{file_id}.png")
        if not os.path.exists(img_path): img_path = os.path.join(img_dir, f"{file_id}.jpg")
        if not os.path.exists(img_path): continue
        
        gt_points, gt_labels = [], []
        try:
            with open(os.path.join(lbl_dir, f"{file_id}.txt"), 'r') as f:
                for ln in f:
                    p = ln.strip().split()
                    if len(p) < 9: continue
                    poly = np.array(list(map(float, p[:8]))).reshape(4, 2)
                    cid = int(p[8])
                    if cid >= len(CLASSES): continue
                    gt_points.append(poly.mean(0))
                    gt_labels.append(cid)
        except: continue
        
        if len(gt_points) == 0:
            pseudo_data_dict[file_id] = []
            continue
            
        gt_points = np.array(gt_points)
        J = len(gt_points)
        img = cv2.medianBlur(cv2.imread(img_path), 3)
        
        # 1. Stage 1
        s_std = torch.eye(2, device=DEVICE).unsqueeze(0).repeat(J, 1, 1) * CFG['sigma']
        mask_s1 = run_watershed_generation(s_std, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
        cv2.watershed(img, mask_s1)
        
        # 2. Geometry Refine
        sigmas_s2, angles_s2 = analyze_geometry_specific(mask_s1, gt_points, CFG['sigma'], DEVICE, METHOD)
        
        # 3. Stage 2
        mask_s2 = run_watershed_generation(sigmas_s2, gt_points, img.shape, CFG['pos'], CFG['neg'], CFG['ds'], DEVICE)
        cv2.watershed(img, mask_s2)
        
        # 4. Final Measurement (Fixed Drift)
        img_boxes = []
        for idx in range(J):
            py, px = np.where(mask_s2 == idx + 1)
            cx, cy = gt_points[idx]
            label_id = gt_labels[idx]
            
            if len(px) < 3:
                img_boxes.append([cx, cy, 16.0, 16.0, 0.0, label_id])
                total_objs += 1
                continue
                
            m_pts = np.stack([px, py], axis=1).astype(np.float32)
            angle = angles_s2[idx]
            
            # anchor strictly at the GT point to avoid center drift
            center = np.array([cx, cy])
            w_raw, h_raw = get_watershed_wh_original_loss(m_pts, center, angle)
            
            # LED: enforce long-edge alignment
            if h_raw > w_raw:
                w_raw, h_raw = h_raw, w_raw
                angle += (np.pi / 2)
            
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            img_boxes.append([cx, cy, w_raw, h_raw, angle, label_id])
            total_objs += 1
            
        pseudo_data_dict[file_id] = img_boxes

    print(f"Total Pseudo RBoxes: {total_objs} | Saving to {SAVE_PATH}...")
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(pseudo_data_dict, f)
    print("DroneVehicle Done.")

if __name__ == '__main__':
    generate_dronevehicle()

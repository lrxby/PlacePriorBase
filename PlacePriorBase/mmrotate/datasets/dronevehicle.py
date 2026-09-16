# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDataset(BaseDataset):
    """DroneVehicle Dataset (Raw Format: 9 columns).

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_id
    Example: 133 532 160 532 165 461 134 459 4
    """

    METAINFO = {
        # class_id: 0, 1, 2, 3, 4
        'classes': ('car', 'bus', 'truck', 'van', 'freight_car'),
        'palette': [
            (220, 20, 60),  # car (Red)
            (119, 11, 32),  # bus (Dark Red)
            (0, 0, 142),    # truck (Dark Blue)
            (0, 0, 230),    # van (Blue)
            (106, 0, 228)   # freight_car (Purple)
        ]
    }

    def __init__(self,
                 img_suffix: str = 'jpg',
                 diff_thr: int = 100,  # kept for interface compatibility (unused)
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from the raw text files."""
        data_list = []
        
        # ann_file is the folder containing the .txt files
        txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
        
        if len(txt_files) == 0:
            raise ValueError(f'No .txt files found in {self.ann_file}')

        # valid class IDs (0-4)
        valid_cat_ids = set(range(len(self.METAINFO['classes'])))

        for txt_file in txt_files:
            data_info = {}
            # file name without extension as img_id
            img_id = osp.split(txt_file)[1][:-4]
            data_info['img_id'] = img_id
            
            # image file name
            img_name = img_id + f'.{self.img_suffix}'
            data_info['file_name'] = img_name
            data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

            instances = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    
                    # 1. format check: exactly 9 columns (8 coords + 1 class)
                    if len(parts) != 9:
                        continue

                    try:
                        # parse coordinates (first 8)
                        bbox = [float(x) for x in parts[:8]]
                        # parse class ID (9th)
                        label_id = int(parts[8])
                    except ValueError:
                        continue

                    # 2. class ID check
                    if label_id not in valid_cat_ids:
                        continue

                    # 3. filter tiny noisy boxes
                    # to avoid errors in PlacePriorBase
                    xs = bbox[0::2]
                    ys = bbox[1::2]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if w < 1 or h < 1: 
                        continue

                    instance = {
                        'bbox': bbox,
                        'bbox_label': label_id,  # use the raw ID
                        'ignore_flag': 0  # no difficulty levels; all valid
                    }
                    instances.append(instance)

            data_info['instances'] = instances
            data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter images with no ground truths."""
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for data_info in self.data_list:
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index."""
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
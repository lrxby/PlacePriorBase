# # Copyright (c) OpenMMLab. All rights reserved.
# import glob
# import os.path as osp
# from typing import List

# from mmengine.dataset import BaseDataset
# from mmrotate.registry import DATASETS


# @DATASETS.register_module()
# class DroneVehicleDataset(BaseDataset):
#     """DroneVehicle Dataset (Raw Format: 9 columns).

#     Format: x1 y1 x2 y2 x3 y3 x4 y4 class_id
#     Example: 133 532 160 532 165 461 134 459 4
#     """

#     METAINFO = {
#         'classes': ('car', 'bus', 'truck', 'van', 'freight_car'),
#         'palette': [
#             (220, 20, 60),  # car (Red)
#             (119, 11, 32),  # bus (Dark Red)
#             (0, 0, 142),    # truck (Dark Blue)
#             (0, 0, 230),    # van (Blue)
#             (106, 0, 228)   # freight_car (Purple)
#         ]
#     }

#     def __init__(self,
#                  img_suffix: str = 'jpg',
#                  **kwargs) -> None:
#         self.img_suffix = img_suffix
#         self.diff_thr = diff_thr
#         super().__init__(**kwargs)

#     def load_data_list(self) -> List[dict]:
#         """Load annotations from the raw text files."""
#         data_list = []
        
#         txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
        
#         if len(txt_files) == 0:
#             raise ValueError(f'No .txt files found in {self.ann_file}')

#         valid_cat_ids = set(range(len(self.METAINFO['classes'])))

#         for txt_file in txt_files:
#             data_info = {}
#             img_id = osp.split(txt_file)[1][:-4]
#             data_info['img_id'] = img_id
            
#             img_name = img_id + f'.{self.img_suffix}'
#             data_info['file_name'] = img_name
#             data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

#             instances = []
#             with open(txt_file, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     parts = line.strip().split()
                    
#                     if len(parts) != 9:
#                         continue

#                     try:
#                         bbox = [float(x) for x in parts[:8]]
#                         label_id = int(parts[8])
#                     except ValueError:
#                         continue

#                     if label_id not in valid_cat_ids:
#                         continue

#                     xs = bbox[0::2]
#                     ys = bbox[1::2]
#                     w = max(xs) - min(xs)
#                     h = max(ys) - min(ys)
#                     if w < 1 or h < 1: 
#                         continue

#                     instance = {
#                         'bbox': bbox,
#                     }
#                     instances.append(instance)

#             data_info['instances'] = instances
#             data_list.append(data_info)

#         return data_list

#     def filter_data(self) -> List[dict]:
#         """Filter images with no ground truths."""
#         if self.test_mode:
#             return self.data_list

#         filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
#             if self.filter_cfg is not None else False

#         valid_data_infos = []
#         for data_info in self.data_list:
#             if filter_empty_gt and len(data_info['instances']) == 0:
#                 continue
#             valid_data_infos.append(data_info)

#         return valid_data_infos

#     def get_cat_ids(self, idx: int) -> List[int]:
#         """Get category ids by index."""
#         instances = self.get_data_info(idx)['instances']
#         return [instance['bbox_label'] for instance in instances]

# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List
import json

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DroneVehicleDataset(BaseDataset):
    """DroneVehicle Dataset (Raw Format: 9 columns).

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_id
    Example: 133 532 160 532 165 461 134 459 4
    """

    METAINFO = {
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
                 diff_thr: int = 100,
                 **kwargs) -> None:
        self.img_suffix = img_suffix
        self.diff_thr = diff_thr
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from the raw text files or json pseudo labels."""
        data_list = []
        
        # ========================================================
        # ========================================================
        if self.ann_file.endswith('.json'):
            with open(self.ann_file, 'r') as f:
                root = json.loads(f.read())

            instances = {}
            for item in root:
                img_id = item['image_id']
                if img_id not in instances.keys():
                    instances[img_id] = []
                instances[img_id].append({
                    'bbox': item['bbox'],
                    'bbox_label': item['category_id'],
                    'ignore_flag': 0
                })

            for img_id in instances.keys():
                data_info = {}
                data_info['img_id'] = str(img_id)
                img_name = str(img_id) + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                img_path_prefix = self.data_prefix.get('img_path', '') if self.data_prefix is not None else ''
                data_info['img_path'] = osp.join(img_path_prefix, img_name)
                data_info['instances'] = instances[img_id]
                data_list.append(data_info)
                
            return data_list

        # ========================================================
        # ========================================================
        txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
        
        if len(txt_files) == 0:
            raise ValueError(f'No .txt files found in {self.ann_file}')

        valid_cat_ids = set(range(len(self.METAINFO['classes'])))

        for txt_file in txt_files:
            data_info = {}
            img_id = osp.split(txt_file)[1][:-4]
            data_info['img_id'] = img_id
            
            img_name = img_id + f'.{self.img_suffix}'
            data_info['file_name'] = img_name
            
            img_path_prefix = self.data_prefix.get('img_path', '') if self.data_prefix is not None else ''
            data_info['img_path'] = osp.join(img_path_prefix, img_name)

            instances = []
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    
                    if len(parts) != 9:
                        continue

                    try:
                        bbox = [float(x) for x in parts[:8]]
                        label_id = int(parts[8])
                    except ValueError:
                        continue

                    if label_id not in valid_cat_ids:
                        continue

                    xs = bbox[0::2]
                    ys = bbox[1::2]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if w < 1 or h < 1: 
                        continue

                    instance = {
                        'bbox': bbox,
                        'bbox_label': label_id,
                        'ignore_flag': 0
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
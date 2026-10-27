# # Copyright (c) OpenMMLab. All rights reserved.
# import glob
# import os.path as osp
# from typing import List
# import json

# from mmengine.dataset import BaseDataset
# from mmrotate.registry import DATASETS


# @DATASETS.register_module()
# class CODRONEDataset(BaseDataset):
#     """CODrone dataset for detection.

#     Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
#     Example: 54.0 156.0 ... 185.0 boat 0
#     """

#     METAINFO = {
#         'classes':
#         ('car', 'truck', 'bus', 'traffic-light',
#          'traffic-sign', 'bridge', 'people', 'bicycle',
#          'motor', 'tricycle', 'boat', 'ship'),
#         # palette is a list of color tuples, which is used for visualization.
#         'palette': [
#             (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
#             (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
#             (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139)
#         ]
#     }

#     def __init__(self,
#                  diff_thr: int = 100,
#                  img_suffix: str = 'jpg',  
#                  **kwargs) -> None:
#         self.diff_thr = diff_thr
#         self.img_suffix = img_suffix
#         super().__init__(**kwargs)

#     def load_data_list(self) -> List[dict]:
#         """Load annotations from an annotation file named as ``self.ann_file``"""
#         cls_map = {c: i for i, c in enumerate(self.metainfo['classes'])}
#         data_list = []

#         if self.ann_file == '':
#             img_files = glob.glob(
#                 osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
#             for img_path in img_files:
#                 data_info = {}
#                 data_info['img_path'] = img_path
#                 img_name = osp.split(img_path)[1]
#                 data_info['file_name'] = img_name
#                 data_info['img_id'] = img_id
                
#                 instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
#                 data_info['instances'] = [instance]
#                 data_list.append(data_info)

#         else:
#             txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
#             if len(txt_files) == 0:
#                 raise ValueError(f'No .txt files found in {self.ann_file}')
            
#             for txt_file in txt_files:
#                 data_info = {}
#                 img_id = osp.split(txt_file)[1][:-4]
#                 data_info['img_id'] = img_id
#                 img_name = img_id + f'.{self.img_suffix}'
#                 data_info['file_name'] = img_name
#                 data_info['img_path'] = osp.join(self.data_prefix['img_path'], img_name)

#                 instances = []
#                 with open(txt_file, 'r') as f:
#                     lines = f.readlines()
#                     for line in lines:
#                         parts = line.strip().split()
                        
#                         if len(parts) < 9:
#                             continue

#                         try:
#                             bbox = [float(x) for x in parts[:8]]
#                             cls_name = parts[8]
#                         except ValueError:
#                             continue

#                         if cls_name not in cls_map:
#                             continue
                        
#                         bbox_label = cls_map[cls_name]

#                         if len(parts) > 9:
#                             try:
#                                 difficulty = int(parts[9])
#                             except ValueError:
#                                 difficulty = 0
#                         else:
#                             difficulty = 0

#                         ignore_flag = 1 if difficulty > self.diff_thr else 0

#                         # ==========================================
#                         # ==========================================
#                         xs = bbox[0::2]
#                         ys = bbox[1::2]
#                         w = max(xs) - min(xs)
#                         h = max(ys) - min(ys)
#                         if w < 1 or h < 1: 
#                             continue

#                         instance = {
#                             'bbox': bbox,
#                             'bbox_label': bbox_label,
#                             'ignore_flag': ignore_flag
#                         }
#                         instances.append(instance)

#                 data_info['instances'] = instances
#                 data_list.append(data_info)

#         return data_list

#     def filter_data(self) -> List[dict]:
#         """Filter annotations according to filter_cfg."""
#         if self.test_mode:
#             return self.data_list

#         filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
#             if self.filter_cfg is not None else False

#         valid_data_infos = []
#         for i, data_info in enumerate(self.data_list):
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
class CODRONEDataset(BaseDataset):
    """CODrone dataset for detection.

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    Example: 54.0 156.0 ... 185.0 boat 0
    """

    METAINFO = {
        'classes':
        ('car', 'truck', 'bus', 'traffic-light',
         'traffic-sign', 'bridge', 'people', 'bicycle',
         'motor', 'tricycle', 'boat', 'ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139)
        ]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',  
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``"""
        cls_map = {c: i for i, c in enumerate(self.metainfo['classes'])}
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
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-len(self.img_suffix)-1]
                data_info['img_id'] = img_id
                
                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

        else:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError(f'No .txt files found in {self.ann_file}')
            
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
                        
                        if len(parts) < 9:
                            continue

                        try:
                            bbox = [float(x) for x in parts[:8]]
                            cls_name = parts[8]
                        except ValueError:
                            continue

                        if cls_name not in cls_map:
                            continue
                        
                        bbox_label = cls_map[cls_name]

                        if len(parts) > 9:
                            try:
                                difficulty = int(parts[9])
                            except ValueError:
                                difficulty = 0
                        else:
                            difficulty = 0

                        ignore_flag = 1 if difficulty > self.diff_thr else 0

                        xs = bbox[0::2]
                        ys = bbox[1::2]
                        w = max(xs) - min(xs)
                        h = max(ys) - min(ys)
                        if w < 1 or h < 1: 
                            continue

                        instance = {
                            'bbox': bbox,
                            'bbox_label': bbox_label,
                            'ignore_flag': ignore_flag
                        }
                        instances.append(instance)

                data_info['instances'] = instances
                data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg."""
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index."""
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
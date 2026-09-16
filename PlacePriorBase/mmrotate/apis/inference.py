# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import List, Optional, Sequence, Union

# import mmcv
# import numpy as np
# import torch
# from mmcv.ops import RoIPool
# from mmcv.transforms import Compose
# from mmdet.structures import DetDataSample, SampleList
# from torch import nn

# from mmrotate.utils import (get_multiscale_patch, get_test_pipeline_cfg,
#                             merge_results_by_nms, slide_window)

# ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


# def inference_detector_by_patches(
#         model: nn.Module,
#         imgs: ImagesType,
#         sizes: List[int],
#         steps: List[int],
#         ratios: List[float],
#         nms_cfg: dict,
#         test_pipeline: Optional[Compose] = None,
#         bs: int = 1) -> Union[DetDataSample, SampleList]:
#     """inference patches with the detector.

#     Split huge image(s) into patches and inference them with the detector.
#     Finally, merge patch results on one huge image by nms.

#     Args:
#         model (nn.Module): The loaded detector.
#         imgs (str, ndarray, Sequence[str/ndarray]): Either image files or
#             loaded images.
#         sizes (list[int]): The sizes of patches.
#         steps (list[int]): The steps between two patches.
#         ratios (list[float]): Image resizing ratios for multi-scale detecting.
#         nms_cfg (dict): nms config.
#         bs (int): Batch size, must greater than or equal to 1.

#     Returns:
#         list[np.ndarray]: Detection results.
#     """
#     assert bs >= 1, 'The batch size must greater than or equal to 1'
#     if isinstance(imgs, (list, tuple)):
#         is_batch = True
#     else:
#         imgs = [imgs]
#         is_batch = False
#     cfg = model.cfg

#     if test_pipeline is None:
#         cfg = cfg.copy()
#         test_pipeline = get_test_pipeline_cfg(cfg)

#         new_test_pipeline = []
#         for pipeline in test_pipeline:
#             if pipeline['type'] != 'LoadAnnotations' and pipeline[
#                     'type'] != 'LoadPanopticAnnotations':
#                 new_test_pipeline.append(pipeline)
#         # set loading pipeline type
#         test_pipeline[0].type = 'LoadPatchFromNDArray'
#         test_pipeline = Compose(new_test_pipeline)

#     if model.data_preprocessor.device.type == 'cpu':
#         for m in model.modules():
#             assert not isinstance(
#                 m, RoIPool
#             ), 'CPU inference with RoIPool is not supported currently.'

#     result_list = []
#     for img in imgs:
#         if not isinstance(img, np.ndarray):
#             img = mmcv.imread(img)
#         height, width = img.shape[:2]
#         sizes, steps = get_multiscale_patch(sizes, steps, ratios)
#         patches = slide_window(width, height, sizes, steps)

#         results = []
#         start = 0
#         while True:
#             # prepare patch data
#             patch_datas = dict(inputs=[], data_samples=[])
#             end = min(start + bs, len(patches))
#             for patch in patches[start:end]:
#                 data_ = dict(
#                     img=img, img_id=0, img_path=None, patch=patch.tolist())
#                 data = test_pipeline(data_)
#                 patch_datas['inputs'].append(data['inputs'])
#                 patch_datas['data_samples'].append(data['data_samples'])

#             # forward the model
#             with torch.no_grad():
#                 results.extend(model.test_step(patch_datas))

#             if end >= len(patches):
#                 break
#             start += bs

#         result_list.append(
#             merge_results_by_nms(
#                 results,
#                 patches[:, :2],
#                 img_shape=(width, height),
#                 nms_cfg=nms_cfg,
#             ))

#     if is_batch:
#         return result_list
#     else:
#         return result_list[0]

# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from torch import nn

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmrotate.registry import MODELS
from mmrotate.utils import (get_multiscale_patch, get_test_pipeline_cfg,
                            merge_results_by_nms, slide_window)

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def init_detector(config: Union[str, Path, Config],
                  checkpoint: Optional[str] = None,
                  palette: str = 'dota',
                  device: str = 'cuda:0',
                  cfg_options: Optional[dict] = None) -> nn.Module:
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If None, the model will
            not load any checkpoint.
        palette (str): Color palette used for visualization. If None,
            the palette will be determined by the dataset. Defaults to 'dota'.
        device (str): Representative of the device. Defaults to 'cuda:0'.
        cfg_options (dict, optional): Options to override some settings in
            the used config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None

    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'dataset_meta' in checkpoint.get('meta', {}):
            model.dataset_meta = checkpoint['meta']['dataset_meta']
        else:
            model.dataset_meta = {'classes': palette}

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model: nn.Module,
                       imgs: ImagesType,
                       test_pipeline: Optional[Compose] = None) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]): Either image files or
            loaded images.
        test_pipeline (:obj:`Compose`, optional): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in incorrect input data type
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_path.
            data_ = dict(img=img, img_id=0, img_path=None)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        
        # build the data pipeline
        data_ = test_pipeline(data_)
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)[0]
            result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def inference_detector_by_patches(
        model: nn.Module,
        imgs: ImagesType,
        sizes: List[int],
        steps: List[int],
        ratios: List[float],
        nms_cfg: dict,
        test_pipeline: Optional[Compose] = None,
        bs: int = 1) -> Union[DetDataSample, SampleList]:
    """inference patches with the detector.

    Split huge image(s) into patches and inference them with the detector.
    Finally, merge patch results on one huge image by nms.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]): Either image files or
            loaded images.
        sizes (list[int]): The sizes of patches.
        steps (list[int]): The steps between two patches.
        ratios (list[float]): Image resizing ratios for multi-scale detecting.
        nms_cfg (dict): nms config.
        bs (int): Batch size, must greater than or equal to 1.

    Returns:
        list[np.ndarray]: Detection results.
    """
    assert bs >= 1, 'The batch size must greater than or equal to 1'
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False
    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)

        new_test_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] != 'LoadAnnotations' and pipeline[
                    'type'] != 'LoadPanopticAnnotations':
                new_test_pipeline.append(pipeline)
        # set loading pipeline type
        test_pipeline[0].type = 'LoadPatchFromNDArray'
        test_pipeline = Compose(new_test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        if not isinstance(img, np.ndarray):
            img = mmcv.imread(img)
        height, width = img.shape[:2]
        sizes, steps = get_multiscale_patch(sizes, steps, ratios)
        patches = slide_window(width, height, sizes, steps)

        results = []
        start = 0
        while True:
            # prepare patch data
            patch_datas = dict(inputs=[], data_samples=[])
            end = min(start + bs, len(patches))
            for patch in patches[start:end]:
                data_ = dict(
                    img=img, img_id=0, img_path=None, patch=patch.tolist())
                data = test_pipeline(data_)
                patch_datas['inputs'].append(data['inputs'])
                patch_datas['data_samples'].append(data['data_samples'])

            # forward the model
            with torch.no_grad():
                results.extend(model.test_step(patch_datas))

            if end >= len(patches):
                break
            start += bs

        result_list.append(
            merge_results_by_nms(
                results,
                patches[:, :2],
                img_shape=(width, height),
                nms_cfg=nms_cfg,
            ))

    if is_batch:
        return result_list
    else:
        return result_list[0]
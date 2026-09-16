# # Copyright (c) OpenMMLab. All rights reserved.
# from .inference import inference_detector_by_patches

# __all__ = ['inference_detector_by_patches']

# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector, inference_detector_by_patches, init_detector

__all__ = ['inference_detector', 'inference_detector_by_patches', 'init_detector']
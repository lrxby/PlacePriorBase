# Copyright (c) OpenMMLab. All rights reserved.
from .gaussian_dist_loss import GDLoss
from .placepriorbase_loss import (VoronoiWatershedLoss,
                                  AnglePriorLoss, SizePriorLoss,
                                  VWRBoxLoss)

__all__ = [
    'GDLoss', 'VoronoiWatershedLoss', 'AnglePriorLoss',
    'SizePriorLoss', 'VWRBoxLoss'
]

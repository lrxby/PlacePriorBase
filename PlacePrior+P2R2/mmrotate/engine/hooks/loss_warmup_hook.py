# Copyright (c) OpenMMLab. All rights reserved.
"""Hook to update current_epoch for AnglePriorLoss warmup."""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class LossWarmupHook(Hook):
    """Update AnglePriorLoss.current_epoch before each training epoch.

    This enables linear warmup of the angle loss weight over the first
    `warmup_epochs` epochs, preventing premature saturation of the
    consistency loss.
    """

    def before_train_epoch(self, runner) -> None:
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        bbox_head = getattr(model, 'bbox_head', None)
        if bbox_head is None:
            return
        loss_angle = getattr(bbox_head, 'loss_angle', None)
        if loss_angle is not None and hasattr(loss_angle, 'current_epoch'):
            loss_angle.current_epoch = runner.epoch
            if runner.rank == 0:
                runner.logger.info(
                    f'[LossWarmupHook] epoch={runner.epoch}, '
                    f'angle warmup weight='
                    f'{min(1.0, (runner.epoch + 1) / loss_angle.warmup_epochs):.3f}'
                )

# Copyright (c) OpenMMLab. All rights reserved.
"""Pseudo-box geometry synchronization transform.

Synchronizes offline pseudo rboxes (``pseudo_boxes``) with the actual
geometric transforms applied to the image by mmdet's ``Resize`` and
``RandomFlip``.

WHY THIS IS NEEDED
------------------
``pseudo_boxes`` is a plain numpy array (loaded by ``LoadPseudoAnnotations``),
NOT an instance of ``BaseBoxes``.  In MMDetection 3.3.0 the ``Resize`` and
``RandomFlip`` transforms only touch ``results['gt_bboxes']`` (which is a
``BaseBoxes`` object) and never iterate over ``bbox_fields``.  As a result,
after ``Resize`` + ``RandomFlip`` the pseudo boxes keep the *pre-augmentation*
coordinates and angle while the image has been resized/flipped.  This makes the
only absolute-direction supervision (``loss_ourwater``) disagree with the image
content on ~75% of the augmented samples (all flipped ones).

This transform compensates for exactly one ``Resize`` + one ``RandomFlip``.
It MUST be placed AFTER ``RandomFlip`` and BEFORE ``PackDetInputs`` in the
train pipeline.

SUPPORTED PIPELINE (strict)
---------------------------
    LoadImageFromFile -> LoadAnnotations -> LoadPseudoAnnotations
    -> ConvertBoxType -> ConvertWeakSupervision
    -> Resize -> RandomFlip -> PseudoBoxSync -> PackDetInputs

* The transform raises ``NotImplementedError`` when the actual ``scale_factor``
  is non-uniform (``|sx - sy| > rel_tol``): rotating boxes do NOT transform
  trivially under non-uniform scaling, and we refuse to silently approximate.
* A guard flag ``_pseudo_synced`` prevents the same pseudo boxes from being
  synchronized twice (e.g. if this transform is accidentally reused in a
  pipeline with a second Resize/RandomFlip stage).
* This transform is NOT valid for pipelines containing RandomRotate, Crop,
  multiple Resize, etc.  The new experiment configs only contain Resize+Flip;
  do not reuse it in other pipelines without extending it.
"""
import numpy as np
from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PseudoBoxSync:
    """Sync ``pseudo_boxes`` with one Resize + one RandomFlip.

    Args:
        rel_tol (float): Relative tolerance on ``|sx - sy|`` before raising
            ``NotImplementedError`` for non-uniform scaling.
    """

    def __init__(self, rel_tol: float = 1e-3):
        self.rel_tol = rel_tol

    def _sync_resize(self, pseudo_boxes, results):
        """Apply the same scaling as ``Resize``.

        ``results['scale_factor']`` in MMDetection 3.3.0 is
        ``(scale_w, scale_h)`` (``new_w/w, new_h/h``).  With ``keep_ratio=True``
        and identical input/output aspect ratio the two values are equal;
        rounding of integer sizes can however introduce tiny differences, so we
        check the *actual* values rather than trusting ``keep_ratio``.
        """
        sf = results.get('scale_factor', None)
        if sf is None:
            return pseudo_boxes
        sx, sy = float(sf[0]), float(sf[1])
        if abs(sx - sy) > self.rel_tol * max(1.0, sx, sy):
            raise NotImplementedError(
                f'PseudoBoxSync: non-uniform scale_factor sx={sx:.6f} '
                f'sy={sy:.6f} (|sx-sy|={abs(sx - sy):.2e} > rel_tol='
                f'{self.rel_tol}). Rotating-box pseudo labels are not '
                'trivially transformable under non-uniform scaling; refusing '
                'to silently approximate. Convert the pipeline to uniform '
                'scaling or extend the transform.')
        out = pseudo_boxes.copy()
        out[:, 0] *= sx  # cx
        out[:, 1] *= sy  # cy
        out[:, 2] *= sx  # w
        out[:, 3] *= sy  # h
        return out

    @staticmethod
    def _wrap_le90(theta):
        """Wrap angle to [-pi/2, pi/2) (le90 convention)."""
        return (theta + np.pi / 2.0) % np.pi - np.pi / 2.0

    def _sync_flip(self, pseudo_boxes, results):
        """Apply the same flip as ``RandomFlip``.

        Mirrors the conventions of ``mmrotate.structures.bbox.RotatedBoxes.
        flip_``:
            horizontal: x = W - x, theta = -theta
            vertical:   y = H - y, theta = -theta
            diagonal:   x = W - x, y = H - y, theta unchanged (mod pi)
        The image shape does not change on flip; ``results['img_shape']``
        (H, W) is already the post-Resize shape.
        """
        if not results.get('flip', False):
            return pseudo_boxes
        direction = results.get('flip_direction', 'horizontal')
        img_shape = results['img_shape']  # (H, W)
        img_w, img_h = img_shape[1], img_shape[0]
        out = pseudo_boxes.copy()
        if direction == 'horizontal':
            out[:, 0] = img_w - out[:, 0]
            out[:, 4] = -out[:, 4]
        elif direction == 'vertical':
            out[:, 1] = img_h - out[:, 1]
            out[:, 4] = -out[:, 4]
        elif direction == 'diagonal':
            out[:, 0] = img_w - out[:, 0]
            out[:, 1] = img_h - out[:, 1]
            # theta unchanged (mod pi)
        else:
            raise NotImplementedError(
                f'PseudoBoxSync: unsupported flip_direction={direction!r}')
        out[:, 4] = self._wrap_le90(out[:, 4])
        return out

    def __call__(self, results):
        if results.get('_pseudo_synced', False):
            raise RuntimeError(
                'PseudoBoxSync: pseudo boxes were already synchronized. '
                'This transform supports exactly one Resize + one RandomFlip '
                'stage; remove the duplicate transform.')
        pseudo_boxes = results.get('pseudo_boxes', None)
        if pseudo_boxes is None or len(pseudo_boxes) == 0:
            results['_pseudo_synced'] = True
            return results
        pseudo_boxes = np.asarray(pseudo_boxes, dtype=np.float32)
        # order matters: first scale, then flip using the scaled image shape
        pseudo_boxes = self._sync_resize(pseudo_boxes, results)
        pseudo_boxes = self._sync_flip(pseudo_boxes, results)
        results['pseudo_boxes'] = pseudo_boxes
        results['_pseudo_synced'] = True
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(rel_tol={self.rel_tol})'

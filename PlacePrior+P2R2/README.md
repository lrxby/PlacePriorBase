<p align="center">
  <h1 align="center">PlacePrior + Point2RBox-v2</h1>
  <p align="center">
    Place-prior losses on top of Point2RBox-v2
  </p>
  <p align="center">
    Runxiang Liu / Kaikai Xie / Qianxi Cao / Yuming Fang / Jiebin Yan / Junjie Chen
  </p>
  <p align="center">
    <i>European Conference on Computer Vision (ECCV) 2026</i>
  </p>
</p>

This folder contains **PlacePrior + P2R2**: the point-supervised pipeline of
[point2rbox-v2](https://github.com/VisionXLab/point2rbox-v2) with the three place-prior losses of
PlacePriorBase added on top.

| Loss | Role |
|---|---|
| `AnglePriorLoss` | predicts the object angle from the point center via a polar coder |
| `SizePriorLoss` | regresses object width/height in L2 (MSE) form |
| `VWRBoxLoss` | supervises the predicted box with the placement prior from the watershed map |

The three losses consume watershed pseudo boxes, so training reads a pseudo-label `.pkl` through
`LoadPseudoAnnotations`, and `PseudoBoxSync` keeps those boxes consistent with resize/flip.

Environment setup, data preparation and pseudo-label generation are shared with the baseline and
are described in the [repository README](../README.md).

> Note: if another `mmrotate` is already installed in the environment, remove it first
> (`pip uninstall -y mmrotate`) before `pip install -e .`; otherwise `tools/*.py` will import
> that copy instead of the bundled one.

## Training and Testing

```bash
# training, single GPU
python tools/train.py configs/placepriorp2r2_dota1.py

# training, multiple GPUs
bash tools/dist_train.sh configs/placepriorp2r2_dota1.py 8

# mAP on the trainval split
python tools/test.py configs/placepriorp2r2_dota1_val.py \
    work_dirs/placepriorp2r2_dota1/epoch_12.pth

# submission package for the DOTA server (writes ./work_dirs/placepriorp2r2_dota1/Task1)
python tools/test.py configs/placepriorp2r2_dota1.py \
    work_dirs/placepriorp2r2_dota1/epoch_12.pth
```

## Released Configs

| Config | Purpose | Dataset | Classes | Epochs |
|---|---|---|---|---|
| `configs/placepriorp2r2_dota1.py` | training + submission package | DOTA-v1.0 | 15 | 12 |
| `configs/placepriorp2r2_dota1_val.py` | mAP on the trainval split | DOTA-v1.0 | 15 | - |

Key settings of the training config: `ss_prob=[0.35, 0.05, 0.6]`, `copy_paste_start_epoch=6`,
milestones `[8, 11]`, AdamW with lr `5e-5`, LinearLR warm-up over the first 500 iterations, and
effective place-prior weights of `0.2 / 0.1 / 0.5` for angle / size / ourwater.

## Loss Switches

The three place-prior losses are controlled by `use_size_loss`, `use_angle_loss` and
`use_ourwater_loss` in `bbox_head`; their effective weight is `loss_weight * scale_factor`.

All three default to `False`, so the head behaves exactly like the original Point2RBox-v2 head
unless a config turns them on. The released DOTA-v1.0 config turns all three on.

## Results

| Dataset | Backbone | mAP (%) |
|---|---|---|
| DOTA-v1.0 | ResNet-50 | 52.63 |

Numbers are reported in the paper (end-to-end training and testing with point annotations only).
Per-class breakdowns and ablation results are given in the paper.

## Acknowledgement

This codebase builds on MMRotate, MMDetection, MMEngine and
[point2rbox-v2](https://github.com/VisionXLab/point2rbox-v2). We thank the authors for their
excellent open-source projects.

## License

Apache 2.0 license.

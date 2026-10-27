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

This folder contains the **PlacePrior + P2R2** model: it keeps the point-supervised pipeline of
[point2rbox-v2](https://github.com/VisionXLab/point2rbox-v2) and adds the three place-prior
losses introduced in PlacePriorBase:

- **Angle prior loss** -- predicts the object angle from the point center via a polar coder.
- **Size prior loss** -- regresses object width/height in L2 (MSE) form.
- **Voronoi Watershed RBox loss** -- supervises the predicted box with the placement prior
  derived from the watershed map.

The three losses consume the pseudo boxes produced by the watershed pipeline, so training reads a
pseudo-label `.pkl` through `LoadPseudoAnnotations`, and `PseudoBoxSync` keeps those boxes
consistent with the resize/flip augmentation. Pseudo-label generation is shared with
`../PlacePriorBase` and is **not** duplicated here.

## Installation

Same environment as `../PlacePriorBase`. The bundled `mmrotate` is self-contained (no separate
`pip install mmrotate` needed).

```bash
# 1. create environment
conda create -n placepriorp2r2 python=3.9 -y
conda activate placepriorp2r2

# 2. install PyTorch (match your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. install mm dependencies (see requirements/mminstall.txt)
pip install -U openmim
mim install mmcv==2.1.0
pip install mmdet==3.3.0 mmengine==0.10.7

# 4. install this repo
pip install -r requirements/runtime.txt
pip install -e .
```

## Getting Started

Training and testing commands are provided in [`configs/README.md`](configs/README.md).
All configs use relative paths and expect the datasets under `data/`.

## Data Preparation

Identical to `../PlacePriorBase` (1024x1024 crops, 200-px overlap; DroneVehicle ships at
840x712 and needs no cropping). Organize each dataset as:

```
data/
├── split_ss_dota/            # DOTA-v1.0
│   ├── trainval/  {images, annfiles}
│   └── test/      {images, annfiles}   # test annotations not public
```

## Pseudo-Label Generation

Use the scripts shipped in `../PlacePriorBase/tools/pseudo_label_generation/`. Each generated
`.pkl` is a dict `{image_id: [[cx, cy, w, h, angle, cls_id], ...]}`; the training config loads it
via `LoadPseudoAnnotations`. The released DOTA-v1.0 config expects

```
data/split_ss_dota/trainval/dota1-rect.pkl
```

## Training

```bash
# single GPU
python tools/train.py configs/placepriorp2r2_dota1.py

# multiple GPUs (8 cards)
bash tools/dist_train.sh configs/placepriorp2r2_dota1.py 8
```

The config uses 12 epochs, AdamW with lr 5e-5, LinearLR(500 iters) followed by
MultiStepLR at epochs 8 and 11, and evaluates mAP on the trainval split every epoch.

## Results

All models are trained end-to-end with **point annotations only**; no box annotations are used.

| Dataset | Backbone | mAP (%) |
|---|---|---|
| DOTA-v1.0 | ResNet-50 | **52.86** |

Notes:

- 52.86 is the best run of the released `placepriorp2r2_dota1.py` config; an independent repeat
  gives 52.13, so expect roughly +/- 0.4 mAP run-to-run.
- The paper reports 52.63 for this setting.
- Configs for DOTA-v1.5, DroneVehicle and CODrone will be added once their runs are finalized.

## Acknowledgement

This codebase builds on MMRotate, MMDetection, MMEngine and
[point2rbox-v2](https://github.com/VisionXLab/point2rbox-v2). We thank the authors for their
excellent open-source projects.

## Citation

```bibtex
@inproceedings{placepriorbase2026,
  title={A Simple Baseline with Placement Prior for Point-Supervised Oriented Object Detection},
  author={Runxiang Liu, Kaikai Xie, Qianxi Cao, Yuming Fang, Jiebin Yan, and Junjie Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

Apache 2.0 license.

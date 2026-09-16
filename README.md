<p align="center">
  <h1 align="center">PlacePriorBase</h1>
  <p align="center">
    Point-Supervised Oriented Object Detection with Placement Prior
  </p>
  <p align="center">
    Runxiang Liu / Kaikai Xie / Qianxi Cao / Yuming Fang / Jiebin Yan / Junjie Chen
  </p>
  <p align="center">
    <i>European Conference on Computer Vision (ECCV) 2026</i>
  </p>
</p>

**PlacePriorBase** is a simple yet effective baseline for **point-supervised oriented object
detection**: given only one point (the GT center) per object, the detector learns to predict
oriented bounding boxes (OBBs) end-to-end, without any box annotations during training.

## Introduction

We rethink point-supervised oriented object detection from the perspective of a **placement
prior**: each object's bounding box is anchored at the GT point (object center), and the geometry
of the box is recovered from a two-stage watershed procedure on the placement prior map, with the
angle extracted from the minimum-area rectangle (minAreaRect) of the watershed region. All later
measurement is strictly anchored to the GT center, avoiding center drift.

Three auxiliary losses are designed for point supervision:

- **Angle prior loss** -- predicts the object angle from the point center via a polar coder.
- **Size prior loss** -- regresses object width/height in L2 (MSE) form.
- **Voronoi Watershed RBox loss** -- supervises the predicted box with the placement prior
  derived from the watershed map.

The baseline is trainable end-to-end on four aerial/drone datasets -- DOTA-v1.0, DOTA-v1.5,
DroneVehicle and CODrone -- with only point annotations.

## Installation

Reference environment: Python 3.9+ / PyTorch 2.x / CUDA 12.x. The codebase is verified with
`torch 2.x`, `mmcv 2.1.0`, `mmdet 3.3.0` and `mmengine 0.10.7` (see `requirements/mminstall.txt`
for the version ranges).

```bash
# 1. create environment
conda create -n placepriorbase python=3.9 -y
conda activate placepriorbase

# 2. install PyTorch (match your CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. install mm dependencies (see requirements/mminstall.txt)
pip install -U openmim
mim install mmcv==2.1.0
pip install mmdet==3.3.0 mmengine==0.10.7

# 4. install the baseline repo (mmrotate is bundled and patched inside the repo)
pip install -r requirements/runtime.txt
pip install -e .
```

The bundled `mmrotate` is self-contained (no separate `pip install mmrotate` needed). The only
modification vs. upstream is a small fix in `mmrotate/evaluation/functional/mean_ap.py` for
evaluating images without ground-truth boxes.

## Getting Started

Training and testing instructions (single-GPU / multi-GPU commands, config list and outputs) are
provided in [`configs/README.md`](configs/README.md).

## Data Preparation

Organize each dataset as follows (the pseudo-label scripts look for `images/` plus `annfiles/` automatically):

```
DOTA-v1.0      split_ss_dota/trainval/          {images/, annfiles/}
DOTA-v1.5      split_ss_dotav1.5/trainval/      {images/, annfiles/}
DroneVehicle   split_ss_dronevehicle/           {trainval/{images,annfiles}, test/{images,annfiles}}
CODrone        split_ss_codrone/trainval/       {images/, annfiles/}
```

All configs use **relative paths** by default: `data_root` points to `data/<dataset>/` relative to
the repository root. The simplest setup is to put your datasets under a `data/` folder at the repo root:

```
repo_root/
└── data/
    ├── split_ss_dota/            # DOTA-v1.0
    │   └── trainval/  {images, annfiles}
    ├── split_ss_dotav1.5/        # DOTA-v1.5
    │   └── trainval/  {images, annfiles}
    ├── split_ss_dronevehicle/    # DroneVehicle
    │   └── {trainval/{images, annfiles}, test/{images, annfiles}}
    └── split_ss_codrone/         # CODrone
        └── trainval/  {images, annfiles}
```

Otherwise, edit `data_root` in `configs/_base_/datasets/*.py` (or the per-dataset `data_root` /
`pkl_path` in `configs/placepriorbase_*.py`) and the `DATA_ROOT` / `SAVE_PATH` variables in
`tools/pseudo_label_generation/*.py` to match your actual dataset locations.

## Pseudo-Label Generation

Pseudo labels are generated **from the GT points only** (one point per object), using a
two-stage watershed pipeline with **minAreaRect angle extraction (Rect method)**. The final box
is measured strictly around the **GT point center**.

Run one script per dataset (edit `DATA_ROOT` / `DEVICE` inside the script if needed):

```bash
python tools/pseudo_label_generation/gen_pseudo_dota1.py
python tools/pseudo_label_generation/gen_pseudo_dota15.py
python tools/pseudo_label_generation/gen_pseudo_dronevehicle.py
python tools/pseudo_label_generation/gen_pseudo_codrone.py
```

Each generated `.pkl` is a dict `{image_id: [[cx, cy, w, h, angle, cls_id], ...]}` and is loaded
by the corresponding training config via `LoadPseudoAnnotations` (the `pkl_path` in each config
already points to the generated file).

## Results

The baseline is trained end-to-end with **point annotations only** (one GT point per object),
no box annotations. All models follow the released 12-epoch configs; the numbers below are the
results reported in the paper.

| Dataset | mAP (%) |
|---|---|
| DOTA-v1.0 | 42.05 |
| DOTA-v1.5 | 32.38 |
| DroneVehicle | 40.75 |
| CODrone | 21.70 |

See the paper for per-class breakdowns and ablations.

## Acknowledgement

The codebase is built on MMRotate, MMDetection and MMEngine. We thank the authors for their
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

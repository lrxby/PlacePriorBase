# Configs

All configs are runnable from the repository root. Datasets are expected under `data/`; see the
main README for the directory layout.

## Available configs

| Config | Purpose | Dataset | Model | Epochs |
|---|---|---|---|---|
| [`placepriorp2r2_dota1.py`](placepriorp2r2_dota1.py) | training + submission package | DOTA-v1.0 (15 classes) | PlacePrior + Point2RBox-v2 | 12 |
| [`placepriorp2r2_dota1_val.py`](placepriorp2r2_dota1_val.py) | mAP on the trainval split | DOTA-v1.0 | same weights | - |

## Commands

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

## Notes

- Every config reads a pseudo-label `.pkl` via `LoadPseudoAnnotations`; generate it with the
  scripts in `../PlacePriorBase/tools/pseudo_label_generation/`.
- The three place-prior losses are controlled by `use_size_loss`, `use_angle_loss` and
  `use_ourwater_loss` in `bbox_head`. Their effective weight is `loss_weight * scale_factor`.
- With those three switches set to `False` (the default in the head), the model is exactly the
  original Point2RBox-v2, so official Point2RBox-v2 configs keep running unchanged.

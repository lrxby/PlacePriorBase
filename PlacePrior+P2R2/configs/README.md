# Configs

All configs are runnable from the repository root. Datasets are expected under `data/`; see the
main README for the directory layout.

## Available configs

| Config | Dataset | Model | Epochs |
|---|---|---|---|
| [`placepriorp2r2_dota1.py`](placepriorp2r2_dota1.py) | DOTA-v1.0 (15 classes) | PlacePrior + Point2RBox-v2 | 12 |

Configs for DOTA-v1.5, DroneVehicle and CODrone will be added once their runs are finalized.

## Commands

```bash
# single GPU
python tools/train.py configs/placepriorp2r2_dota1.py

# multiple GPUs
bash tools/dist_train.sh configs/placepriorp2r2_dota1.py 8

# evaluation on the trainval split
python tools/test.py configs/placepriorp2r2_dota1.py work_dirs/placepriorp2r2_dota1/epoch_12.pth

# submission package for the DOTA server
python tools/test.py configs/placepriorp2r2_dota1.py work_dirs/placepriorp2r2_dota1/epoch_12.pth \
    --cfg-options test_evaluator=test_evaluator_submission
```

## Notes

- Every config reads a pseudo-label `.pkl` via `LoadPseudoAnnotations`; generate it with the
  scripts in `../PlacePriorBase/tools/pseudo_label_generation/`.
- The three place-prior losses are controlled by `use_size_loss`, `use_angle_loss` and
  `use_ourwater_loss` in `bbox_head`. Their effective weight is `loss_weight * scale_factor`.
- Baseline behaviour of Point2RBox-v2 is preserved: with those three switches set to `False`
  (the default in the head), the model is exactly the original Point2RBox-v2.

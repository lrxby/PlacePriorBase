# Training / Testing Instructions

All commands below are run from the repository root `PlacePriorBase/`.

## Configs

| Config | Dataset | Classes | Epochs |
|---|---|---|---|
| placepriorbase_dota1.py | DOTA-v1.0 | 15 | 12 |
| placepriorbase_dota15.py | DOTA-v1.5 | 16 | 12 |
| placepriorbase_dronevehicle.py | DroneVehicle | 5 | 12 |
| placepriorbase_codrone.py | CODrone | 12 | 12 |

Training recipe (identical for all datasets): **12 epochs**, batch size 2 per GPU, AdamW
(lr=5e-5, wd=0.05), LinearLR warm-up (500 iters, start_factor 1/3) + MultiStepLR
(milestones [8, 11], gamma 0.1). Validation runs every epoch on the training split
(DOTA family / CODrone) or the official test split (DroneVehicle).

## Training

Single GPU:

```bash
python tools/train.py configs/placepriorbase_dota1.py          # DOTA-v1.0
python tools/train.py configs/placepriorbase_dota15.py         # DOTA-v1.5
python tools/train.py configs/placepriorbase_dronevehicle.py   # DroneVehicle
python tools/train.py configs/placepriorbase_codrone.py        # CODrone
```

Multi-GPU:

```bash
bash tools/dist_train.sh configs/placepriorbase_dota1.py 2
```

Custom working directory:

```bash
python tools/train.py configs/placepriorbase_dota1.py --work-dir work_dirs/my_run
```

**Outputs.** Training writes to `work_dirs/<config-name>/`:

- `console.log` -- full training log;
- `epoch_*.pth` -- one checkpoint per epoch;
- the checkpoint with the best validation `dota/mAP` is saved separately and updated
  automatically during training (see `save_best` in `configs/_base_/default_runtime.py`).

## Testing

Evaluate a trained checkpoint on the val/test split:

```bash
python tools/test.py configs/placepriorbase_dota1.py work_dirs/placepriorbase_dota1/epoch_12.pth
```

Multi-GPU:

```bash
bash tools/dist_test.sh configs/placepriorbase_dota1.py work_dirs/placepriorbase_dota1/epoch_12.pth 2
```

For DOTA **test set submission** (test images are not public), run inference with
`format_only=True` to dump prediction `.txt` files, zip them and submit to the DOTA
evaluation server:

```bash
python tools/test.py configs/placepriorbase_dota1.py work_dirs/placepriorbase_dota1/epoch_12.pth \
    --cfg-options test_evaluator.format_only=True test_evaluator.merge_patches=False \
                  test_dataloader.dataset.ann_file=test/annfiles/ test_dataloader.dataset.data_prefix.img_path=test/images/
```

## Resume from a checkpoint

```bash
python tools/train.py configs/placepriorbase_dota1.py --work-dir work_dirs/placepriorbase_dota1 --resume
```

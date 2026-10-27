# ---------------------------------------------------------------
# DOTA-v1.0 (15 classes)
# PlacePrior + Point2RBox-v2: place-prior losses (angle / size /
# Voronoi-watershed box) on top of Point2RBox-v2.
# Recipe: 12 epochs, AdamW lr=5e-5, LinearLR(500 iter) + MultiStepLR[8, 11]
custom_imports = dict(
    imports=['mmrotate.datasets.transforms.loading_pseudo',
             'mmrotate.datasets.transforms.pseudo_box_sync'],
    allow_failed_imports=False)

_base_ = [
    '_base_/datasets/dota.py', '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
angle_version = 'le90'

randomness = dict(seed=1, deterministic=False)

model = dict(
    type='Point2RBoxV2',

    ss_prob=[0.35, 0.05, 0.6],
    copy_paste_start_epoch=6,

    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),

    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),

    bbox_head=dict(
        type='Point2RBoxV2Head',
        num_classes=15,
        in_channels=128,
        feat_channels=128,
        strides=[8],

        edge_loss_start_epoch=6,
        joint_angle_start_epoch=1,
        voronoi_type='standard',
        voronoi_thres=dict(
            default=[0.994, 0.005],
            override=(([2, 11], [0.999, 0.6]),
                      ([7, 8, 10, 14], [0.95, 0.005]))),
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},

        angle_coder=dict(
            type='PSCCoder',
            angle_version='le90',
            dual_freq=False,
            num_step=3,
            thr_mod=0),

        # ------------------ loss switches ------------------
        use_bbox_loss=True,
        use_overlap_loss=True,
        use_voronoi_loss=True,
        use_edg_loss=True,
        use_ss_loss=True,
        # place-prior losses
        use_size_loss=True,
        use_angle_loss=True,
        use_ourwater_loss=True,
        # ---------------------------------------------------------------

        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),

        loss_overlap=dict(
            type='GaussianOverlapLoss', loss_weight=10.0, lamb=0),

        loss_voronoi=dict(
            type='VoronoiWatershedLoss', loss_weight=5.0),

        loss_bbox_edg=dict(type='EdgeLoss', loss_weight=0.3),

        loss_ss=dict(
            type='Point2RBoxV2ConsistencyLoss', loss_weight=1.0),

        # place-prior losses (effective weight = loss_weight * scale_factor)
        loss_angle=dict(
            type='AnglePriorLoss',
            loss_weight=0.2,
            scale_factor=1.0,
            eps=0.01,
            topk=0.90,
            k_radius=2.0,
            score_alpha=1.0,
            target_classes=[0, 4, 5, 6, 7, 8, 10, 12, 14]),

        loss_size=dict(
            type='SizePriorLoss',
            loss_weight=0.1,
            scale_factor=1.0,
            loss_type='l2',
            topk=0.90,
            beta=1.0,
            target_classes=[0, 1, 4, 7, 8, 9, 10, 11, 14]),

        loss_ourwater=dict(
            type='VWRBoxLoss',
            loss_weight=0.5,
            scale_factor=1.0,
            topk=0.90)),

    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# train_pipeline:
# LoadPseudoAnnotations -> ConvertBoxType -> ConvertWeakSupervision
#   -> Resize -> RandomFlip -> PseudoBoxSync -> PackDetInputs
# PseudoBoxSync runs after RandomFlip and before PackDetInputs:
#   first scale by (scale_factor), then flip using the scaled img_shape
#   horizontal/vertical: theta -> -theta; diagonal: theta unchanged (mod pi), x/y flipped
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='LoadPseudoAnnotations',
         pkl_path='data/split_ss_dota/trainval/dota1-rect.pkl'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='ConvertWeakSupervision', point_proportion=1., hbox_proportion=0),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PseudoBoxSync',
         rel_tol=1e-3),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'pseudo_boxes', 'pseudo_valid')
    )
]

test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

data_root = 'data/split_ss_dota/'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))

# optimizer: AdamW 5e-5
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00005,
        betas=(0.9, 0.999),
        weight_decay=0.05))

# lr schedule: LinearLR 500 iters + MultiStepLR [8, 11]
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# val: mAP on trainval
val_evaluator = dict(type='DOTAMetric', metric='mAP')

# Note: explicitly disable format_only / merge_patches; otherwise the base
# test_evaluator (format_only=True, merge_patches=True) survives config merge
# and evaluation only produces a submission.
test_evaluator = dict(type='DOTAMetric', metric='mAP',
                      format_only=False, merge_patches=False)

test_evaluator_submission = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/p2r2_dota1/Task1')

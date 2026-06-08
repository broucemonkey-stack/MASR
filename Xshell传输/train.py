auto_scale_lr = dict(base_batch_size=32)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=6,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=0,
        rule='greater',
        save_best='single-label/f1-score',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=6,
        topk=(1, ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, type='Adam', weight_decay=0.0001))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1',
            '3',
            '4',
            '5',
            '7',
            '9',
        ],
        data_root='/home/lxleave/mmpretrain/projects/Blast_UAV/20260608/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                160,
                160,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(
        items=[
            'precision',
            'recall',
            'f1-score',
        ], type='SingleLabelMetric'),
    dict(type='ConfusionMatrix'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        160,
        160,
    ), type='Resize'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1',
            '3',
            '4',
            '5',
            '7',
            '9',
        ],
        data_root='/home/lxleave/mmpretrain/projects/Blast_UAV/20260608/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                160,
                160,
            ), type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes=[
            '1',
            '3',
            '4',
            '5',
            '7',
            '9',
        ],
        data_root='/home/lxleave/mmpretrain/projects/Blast_UAV/20260608/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                160,
                160,
            ), type='Resize'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(topk=(1, ), type='Accuracy'),
    dict(
        items=[
            'precision',
            'recall',
            'f1-score',
        ], type='SingleLabelMetric'),
    dict(type='ConfusionMatrix'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        init_kwargs=dict(name='exp_0605', project='Blast_UAV'),
        type='WandbVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            init_kwargs=dict(name='exp_0605', project='Blast_UAV'),
            type='WandbVisBackend'),
    ])
work_dir = 'work_dirs/blast_uav/train/0608'

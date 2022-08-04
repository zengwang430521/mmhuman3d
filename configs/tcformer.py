_base_ = ['_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(interval=6, metric=['pa-mpjpe', 'mpjpe'])

img_res = 224

# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=5.0e-5),
    head=dict(type='Adam', lr=5.0e-5),
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='Fixed', by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=6)

# model settings
width = 32
downsample = False
use_conv = True


find_unused_parameters = True

model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='TCFormer',
        pretrained='pretrained/tcformer-4e1adbf1_20220421.pth',
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        num_layers=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_path_rate=0.1),
    neck=dict(type='TCGap'),
    head=dict(
        type='HMRHead',
        feat_dim=512,
        smpl_mean_params='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_24',
        model_path='data/body_models/smpl',
        keypoint_approximate=True,
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    convention='smpl_24',
    loss_keypoints3d=dict(type='MSELoss', loss_weight=300),
    loss_keypoints2d=dict(type='MSELoss', loss_weight=150),
    loss_smpl_pose=dict(type='MSELoss', loss_weight=60),
    loss_smpl_betas=dict(type='MSELoss', loss_weight=60 * 0.001),
    loss_camera=dict(type='CameraPriorLoss', loss_weight=1),
)

# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'has_keypoints3d', 'has_keypoints2d', 'smpl_body_pose',
    'smpl_global_orient', 'smpl_betas', 'smpl_transl', 'keypoints2d',
    'keypoints3d', 'sample_idx'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomChannelNoise', noise_factor=0.4),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_24'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_24',
                ann_file='h36m_train_mine.npz'),
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_24',
                ann_file='eft_coco_all.npz'),
            dict(
                type=dataset_type,
                dataset_name='lspet',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_24',
                ann_file='eft_lspet.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_24',
                ann_file='eft_mpii.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_24',
                ann_file='mpi_inf_3dhp_train.npz'),
        ],
        partition=[0.5, 0.233, 0.046, 0.021, 0.2],
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)

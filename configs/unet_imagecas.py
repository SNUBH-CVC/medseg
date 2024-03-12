import os
from datetime import datetime

import numpy as np
import torch
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform, ContrastAugmentationTransform,
    GammaTransform)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform, GaussianNoiseTransform)
from batchgenerators.transforms.resample_transforms import \
    SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import (MirrorTransform,
                                                           SpatialTransform)
from monai.data import DataLoader
from monai.handlers import LrScheduleHandler, MeanDice, StatsHandler
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, RandSpatialCropd,
                              ScaleIntensityRangePercentilesd, Spacingd,
                              SqueezeDimd, adaptor)

from medseg.dataset import ImageCasDataset
from medseg.handlers import MedSegMLFlowHandler

mlflow_tracking_uri = "file:///data/mlruns"
dataset_dir = "/data/imagecas"
roi_size = (256, 256, 128)
cfg_path = __file__
experiment_name = os.path.splitext(os.path.basename(cfg_path))[0]

pixdim = [0.35, 0.35, 0.5]  # median
eval_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim),
        EnsureTyped(keys=["image", "label"]),
    ]
)
eval_post_pred_transforms = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

key_metric_name = "mean_dice"
metrics = {key_metric_name: MeanDice()}
eval_post_label_transforms = Compose([AsDiscrete(threshold=0.5)])
inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=4, overlap=0.25)
evaluator_kwargs = dict(
    metrics=metrics,
    prepare_batch=lambda batch, device, non_blocking: (
        batch["image"].to(device),
        batch["label"].to(device),
    ),
    output_transform=lambda x, y, y_pred: (
        [eval_post_pred_transforms(i) for i in y_pred],
        [eval_post_label_transforms(i) for i in y],
    ),
    model_fn=lambda model, x: inferer(x, model),
)


def prepare_train():
    num_workers = 4
    batch_size = 16
    max_epochs = 400

    rotation_range = (-30.0 / 360 * 2 * np.pi, 30.0 / 360 * 2 * np.pi)
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/ostiapoints_train_tools/ostia_net_data_provider_aug.py
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
            ),
            Spacingd(keys=["image", "label"], pixdim=pixdim),
            NormalizeIntensityd(keys=["image"]),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"], data_type="numpy"),
            # https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation
            adaptor(
                SpatialTransform(
                    roi_size,
                    patch_center_dist_from_border=None,
                    do_elastic_deform=False,
                    alpha=(0, 0),
                    sigma=(0, 0),
                    do_rotation=True,
                    angle_x=rotation_range,
                    angle_y=rotation_range,
                    angle_z=rotation_range,
                    p_rot_per_axis=1,
                    do_scale=True,
                    scale=(0.7, 1.4),
                    border_mode_data="constant",
                    border_cval_data=0,
                    order_data=3,
                    border_mode_seg="constant",
                    border_cval_seg=-1,
                    order_seg=1,
                    random_crop=False,
                    p_el_per_sample=0,
                    p_scale_per_sample=0.2,
                    p_rot_per_sample=0.2,
                    independent_scale_for_each_axis=False,
                    data_key="image",
                    label_key=("label"),
                ),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                GaussianNoiseTransform(p_per_sample=0.1, data_key="image"),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                GaussianBlurTransform(
                    (0.5, 1.0),
                    different_sigma_per_channel=True,
                    p_per_sample=0.2,
                    p_per_channel=0.5,
                    data_key="image",
                ),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                BrightnessMultiplicativeTransform(
                    multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key="image"
                ),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                SimulateLowResolutionTransform(
                    zoom_range=(0.5, 1),
                    per_channel=True,
                    p_per_channel=0.5,
                    order_downsample=0,
                    order_upsample=3,
                    p_per_sample=0.25,
                    ignore_axes=None,
                    data_key="image",
                ),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                GammaTransform(
                    (0.7, 1.5),
                    True,
                    True,
                    retain_stats=True,
                    p_per_sample=0.1,
                    data_key="image",
                ),
                {"image": "image", "label": "label"},
            ),
            adaptor(
                MirrorTransform((0, 1, 2), data_key="image", label_key=("label")),
                {"image": "image", "label": "label"},
            ),
            SqueezeDimd(keys=["image", "label"], dim=0),
        ]
    )
    train_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        mode="train",
        transform=train_transform,
        cache_rate=0.0,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        mode="validation",
        transform=eval_transform,
        cache_rate=0.0,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=3e-5)
    lr_scheduler = LrScheduleHandler(
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9
        )
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_mlflow_handler = MedSegMLFlowHandler(
        mlflow_tracking_uri,
        output_transform=lambda x: x,
        experiment_name=experiment_name,
        run_name=run_name,
        artifacts_at_start={cfg_path: "config"},
        optimizer=optimizer,
    )
    val_mlflow_handler = lambda trainer: MedSegMLFlowHandler(
        mlflow_tracking_uri,
        output_transform=lambda x: None,
        experiment_name=experiment_name,
        run_name=run_name,
        save_dict={"model": model},
        key_metric_name=key_metric_name,
        global_epoch_transform=lambda x: trainer.state.epoch,
        log_model=True,
    )
    train_stats_handler = StatsHandler(
        name="train_log",
        tag_name="train_loss",
        output_transform=lambda x: x,
    )
    val_stats_handler = lambda trainer: StatsHandler(
        name="train_log",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )

    trainer_kwargs = dict(
        prepare_batch=lambda batch, device, non_blocking: (
            batch["image"].to(device),
            batch["label"].to(device),
        ),
    )
    trainer_handlers = [lr_scheduler, train_stats_handler, train_mlflow_handler]
    evaluator_handlers = [val_stats_handler, val_mlflow_handler]

    return dict(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        trainer_kwargs=trainer_kwargs,
        evaluator_kwargs=evaluator_kwargs,
        max_epochs=max_epochs,
        trainer_handlers=trainer_handlers,
        evaluator_handlers=evaluator_handlers,
    )


def prepare_test():
    test_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        mode="test",
        transform=eval_transform,
        cache_rate=0.0,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    evaluation_handlers = []

    return dict(
        dataloader=test_dataloader,
        evaluator_kwargs=evaluator_kwargs,
        evaluator_handlers=evaluation_handlers,
    )

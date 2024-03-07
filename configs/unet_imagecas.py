import os
from datetime import datetime

import torch
from monai.data import DataLoader
from monai.handlers import MeanDice, StatsHandler
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, RandCropByPosNegLabeld,
                              RandScaleIntensityd, RandShiftIntensityd,
                              RandSpatialCropd,
                              ScaleIntensityRangePercentilesd)

from medseg.dataset import ImageCasDataset
from medseg.handler import MedSegMLFlowHandler

mlflow_tracking_uri = "file:///data/mlruns"
dataset_dir = "/data/imagecas"
roi_size = (256, 256, 128)
cfg_path = __file__
experiment_name = os.path.splitext(os.path.basename(cfg_path))[0]

eval_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        NormalizeIntensityd(keys=["image"]),
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
    batch_size = 4
    max_epochs = 500

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/ostiapoints_train_tools/ostia_net_data_provider_aug.py
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
            ),
            NormalizeIntensityd(keys=["image"]),
            # RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size),
            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=roi_size,
            #     pos=1,
            #     neg=1,
            #     num_samples=4,
            #     image_key="image",
            # ),
        ]
    )
    train_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        section="training",
        transform=train_transform,
        cache_rate=0.0,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        section="validation",
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
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_mlflow_handler = MedSegMLFlowHandler(
        mlflow_tracking_uri,
        output_transform=lambda x: x,
        experiment_name=experiment_name,
        run_name=run_name,
        artifacts_at_start={cfg_path: "config"},
    )
    val_mlflow_hanlder = lambda trainer: MedSegMLFlowHandler(
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
    trainer_handlers = [train_stats_handler, train_mlflow_handler]
    evaluator_handlers = [val_stats_handler, val_mlflow_hanlder]

    return dict(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        scheduler=scheduler,
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
        section="test",
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

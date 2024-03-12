import os
from datetime import datetime

import torch
from monai.data import DataLoader
from monai.handlers import LrScheduleHandler, MeanDice, StatsHandler
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.transforms import (Activations, AsDiscrete, Compose,
                              EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              NormalizeIntensityd, RandAdjustContrastd,
                              RandAxisFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandRotated,
                              RandScaleIntensityd, RandSpatialCropd, RandZoomd,
                              ScaleIntensityRangePercentilesd, Spacingd)

from medseg.dataset import ImageCasDataset
from medseg.handlers import MedSegMLFlowHandler
from medseg.losses import clDiceLoss
from medseg.models import SkelUNet

mlflow_tracking_uri = "file:///data/mlruns"
dataset_dir = "/data/imagecas"
roi_size = (128, 128, 64)
cfg_path = __file__
experiment_name = os.path.splitext(os.path.basename(cfg_path))[0]

pixdim = [0.35, 0.35, 0.5]  # median
eval_transform = Compose(
    [
        LoadImaged(keys=["image", "label", "skeleton"]),
        EnsureChannelFirstd(keys=["image", "label", "skeleton"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
        ),
        Spacingd(keys=["image", "label", "skeleton"], pixdim=pixdim),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label", "skeleton"]),
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
        (batch["label"].to(device), batch["skeleton"].to(device)),
    ),
    output_transform=lambda x, y, y_pred: (
        [eval_post_pred_transforms(i) for i in y_pred[0]],
        [eval_post_label_transforms(i) for i in y[0]],
    ),
    model_fn=lambda model, x: inferer(x, model),
)


def prepare_train():
    num_workers = 4
    batch_size = 2
    max_epochs = 400

    range_rotation = (-0.523, 0.523)
    prob = 0.2
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label", "skeleton"]),
            EnsureChannelFirstd(keys=["image", "label", "skeleton"]),
            # https://github.com/BubblyYi/Coronary-Artery-Tracking-via-3D-CNN-Classification/blob/master/ostiapoints_train_tools/ostia_net_data_provider_aug.py
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.05, upper=0.95, b_min=-4000, b_max=4000
            ),
            Spacingd(keys=["image", "label", "skeleton"], pixdim=pixdim),
            NormalizeIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label", "skeleton"]),
            RandSpatialCropd(keys=["image", "label", "skeleton"], roi_size=roi_size),
            # https://github.com/PierreRouge/Cascaded-U-Net-for-vessel-segmentation
            RandRotated(
                ["image", "label", "skeleton"],
                prob=prob,
                range_x=range_rotation,
                range_y=range_rotation,
                range_z=range_rotation,
            ),
            RandZoomd(["image", "label", "skeleton"], prob=prob, min_zoom=0.7, max_zoom=1.4),
            RandGaussianNoised(keys=["image"], prob=prob),
            RandGaussianSmoothd(keys=["image"], prob=prob),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=prob),
            RandAxisFlipd(["image", "label", "skeleton"], prob=prob),
            RandAdjustContrastd(keys=["image"], prob=prob),
        ]
    )
    train_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        mode="train",
        transform=train_transform,
        cache_rate=0.0,
        use_mask=True,
        use_skeleton=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataset = ImageCasDataset(
        dataset_dir=dataset_dir,
        mode="validation",
        transform=eval_transform,
        cache_rate=0.0,
        use_mask=True,
        use_skeleton=True,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = SkelUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        pred_mask=True,
    )
    cldice_loss = clDiceLoss()
    dice_loss = DiceLoss(sigmoid=True)
    alpha = 0.3
    loss_function = lambda y_pred, y: (1 - alpha) * dice_loss(
        y_pred[0], y[0]
    ) + alpha * cldice_loss(y_pred, y)
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
            (batch["label"].to(device), batch["skeleton"].to(device)),
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
        use_mask=False,
        use_skeleton=True,
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

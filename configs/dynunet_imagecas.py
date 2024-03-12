import os
from datetime import datetime

import torch
from monai.data import DataLoader
from monai.handlers import LrScheduleHandler, MeanDice, StatsHandler
from monai.inferers import SlidingWindowInferer
from monai.losses.dice import DiceLoss
from monai.networks.nets import DynUnet
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
roi_size = (128, 128, 64)
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
    batch_size = 8
    max_epochs = 200

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

    deep_supervision = True
    kernels = [[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    model = DynUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=deep_supervision,
        deep_supr_num=3,
    )
    dice_loss = DiceLoss(sigmoid=True)
    if deep_supervision:
        loss_function = lambda input, target: sum(
            0.5**i * dice_loss(p, target)
            for i, p in enumerate(torch.unbind(input, dim=1))
        )
    else:
        loss_function = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
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
    trainer_handlers = [lr_scheduler, train_stats_handler, train_mlflow_handler]
    evaluator_handlers = [val_stats_handler, val_mlflow_hanlder]

    return dict(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        lr_scheduler=lr_scheduler,
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

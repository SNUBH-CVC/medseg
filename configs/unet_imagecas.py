import os
from datetime import datetime

import torch
from monai.handlers import LrScheduleHandler, StatsHandler
from monai.losses.dice import DiceLoss
from monai.networks.nets.unet import UNet

from medseg.handlers import MedSegMLFlowHandler

dataset_cfg_path = "configs/base/datasets/imagecas.py"


def prepare_train(
    train_dataloader,
    val_dataloader,
    eval_post_pred_transforms,
    eval_post_label_transforms,
    inferer,
    metrics,
    key_metric_name,
    mlflow_tracking_uri=None,
    *args,
    **kwargs
):
    max_epochs = 400
    cfg_path = __file__
    experiment_name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    train_mlflow_handler = MedSegMLFlowHandler(
        mlflow_tracking_uri,
        output_transform=lambda x: x,
        experiment_name=experiment_name,
        run_name=run_name,
        artifacts_at_start={cfg_path: "config", dataset_cfg_path: "config"},
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


def prepare_test(
    test_dataloader,
    eval_post_pred_transforms,
    eval_post_label_transforms,
    inferer,
    metrics,
    *args,
    **kwargs
):
    evaluation_handlers = []
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

    return dict(
        dataloader=test_dataloader,
        evaluator_kwargs=evaluator_kwargs,
        evaluator_handlers=evaluation_handlers,
    )

import os
from datetime import datetime

import torch
from monai.data import DataLoader
from monai.handlers import MeanDice, StatsHandler
from monai.losses.dice import DiceLoss
from monai.networks.nets.swin_unetr import SwinUNETR

from medseg.core.utils import get_attributes_from_module
from medseg.handlers import MedSegMLFlowHandler

mlflow_tracking_uri = "file:///data/mlruns"
cfg_path = __file__
experiment_name = os.path.splitext(os.path.basename(cfg_path))[0]
dataset_cfg_path = "configs/datasets/imagecas.py"

dataset_attrs = get_attributes_from_module(dataset_cfg_path)
train_dataset = dataset_attrs["train_dataset"]
val_dataset = dataset_attrs["val_dataset"]
test_dataset = dataset_attrs["test_dataset"]
inferer = dataset_attrs["inferer"]
eval_post_pred_transforms = dataset_attrs["eval_post_pred_transforms"]
eval_post_label_transforms = dataset_attrs["eval_post_label_transforms"]
roi_size = dataset_attrs["roi_size"]

key_metric_name = "mean_dice"
metrics = {key_metric_name: MeanDice()}
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
    batch_size = 2
    max_epochs = 500

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = SwinUNETR(
        img_size=roi_size,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
    )
    model.load_from(weights=torch.load("./pretrained/model_swinvit.pt"))
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    evaluation_handlers = []

    return dict(
        dataloader=test_dataloader,
        evaluator_kwargs=evaluator_kwargs,
        evaluator_handlers=evaluation_handlers,
    )

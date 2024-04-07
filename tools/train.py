import logging

import hydra
import mlflow
import pyrootutils
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from monai.handlers import LrScheduleHandler
from omegaconf import DictConfig

from medseg.core.utils import set_mlflow_tracking_uri

root = pyrootutils.setup_root(
    search_from=__file__,
    pythonpath=True,
)

logger = logging.getLogger(__name__)

HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}


@hydra.main(**HYDRA_PARAMS)
def train(cfg: DictConfig):
    device = cfg.device
    model = hydra.utils.instantiate(cfg.model.obj).to(device)
    if cfg.model.pretrained_weight is not None:
        model.load_from(torch.load(cfg.model.pretrained_weight))
    if cfg.multi_gpu:
        assert device.startswith("cuda")
        if device == "cuda":
            device_ids = [i for i in range(torch.cuda.device_count())]
        else:
            device_ids = [int(i) for i in device.split(":")[1].split(",")]
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=device_ids)

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataloader = dataset["train_dataloader"]
    val_dataloader = dataset["val_dataloader"]
    inferer = dataset["inferer"]
    prepare_batch = dataset["prepare_batch"]
    eval_post_pred_transforms = dataset["eval_post_pred_transforms"]
    eval_post_label_transforms = dataset["eval_post_label_transforms"]
    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = LrScheduleHandler(
        hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    )
    metrics = hydra.utils.instantiate(cfg.metrics)

    trainer_kwargs = dict(
        prepare_batch=prepare_batch,
    )
    evaluator_kwargs = dict(
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (
            [eval_post_pred_transforms(i) for i in y_pred],
            [eval_post_label_transforms(i) for i in y],
        ),
        model_fn=lambda model, x: inferer(x, model),
    )

    trainer_handlers = [lr_scheduler]
    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device, **trainer_kwargs
    )
    for handler in trainer_handlers:
        handler.attach(trainer)

    evaluator = create_supervised_evaluator(model, metrics, device, **evaluator_kwargs)

    @trainer.on(Events.ITERATION_COMPLETED)
    def trainer_iteration_completed(engine):
        mlflow.log_metric(
            "train_loss", value=engine.state.output, step=engine.state.iteration
        )
        logger.info(f"train_loss [{engine.state.iteration}]: {engine.state.output}")
        optimizer_param_names = ["lr"]
        for param_name in optimizer_param_names:
            params = {
                f"{param_name} group_{i}": float(param_group[param_name])
                for i, param_group in enumerate(optimizer.param_groups)
            }
            mlflow.log_metrics(params, step=engine.state.iteration)
            logger.info(f"optimizer params [{engine.state.iteration}]: {params}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_epoch_completed(engine):
        evaluator.run(val_dataloader)

    best_metric = -1

    @evaluator.on(Events.EPOCH_COMPLETED)
    def evaluator_epoch_completed(engine):
        nonlocal best_metric
        for name, value in engine.state.metrics.items():
            if name == cfg.key_metric_name:
                if value > best_metric:
                    best_metric = value
                    mlflow.pytorch.log_model(model, "model")
                    logger.info(
                        f"saved new best metric model [{trainer.state.epoch}]: {best_metric}"
                    )
            mlflow.log_metric(name, value=value, step=trainer.state.epoch)

    set_mlflow_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run() as run:
        # log hydra output directory path to mlflow.
        mlflow.log_param("hydra_output_subdir", HydraConfig.get()["output_subdir"])
        trainer.run(train_dataloader, max_epochs=cfg.max_epochs)


if __name__ == "__main__":
    train()

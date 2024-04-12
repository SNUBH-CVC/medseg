import argparse
import os
import tempfile

import hydra
import mlflow
import omegaconf
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from monai.handlers import LrScheduleHandler

from medseg.core.utils import set_mlflow_tracking_uri, setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydra_config_path", type=str, default="configs/train.yaml")
    parser.add_argument("--mlflow_tracking_uri", default=None)
    parser.add_argument("--resume_run_id", default=None)
    parser.add_argument("--log_path", type=str, default="./out.log")
    return parser.parse_args()


def train():
    args = parse_args()
    logger = setup_logger(args.log_path)
    if args.resume_run_id is not None:
        logger.info(f"resume run_id: {args.resume_run_id}")
        resume = True
        assert args.mlflow_tracking_uri is not None

        set_mlflow_tracking_uri(args.mlflow_tracking_uri)
        client = mlflow.MlflowClient(args.mlflow_tracking_uri)
        with tempfile.TemporaryDirectory() as tmpdirname:
            cfg_download_path = client.download_artifacts(
                args.resume_run_id, "config", dst_path=tmpdirname
            )
            cfg_path = os.path.join(cfg_download_path, "config.yaml")
            cfg = omegaconf.OmegaConf.load(cfg_path)
            checkpoint_download_path = client.download_artifacts(
                args.resume_run_id, "model", dst_path=tmpdirname
            )
            checkpoint_path = os.path.join(checkpoint_download_path, "checkpoint.pth")
            checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        resume_epoch = checkpoint["epoch"]
        best_metric = max(
            i.value
            for i in client.get_metric_history(args.resume_run_id, cfg.key_metric_name)
        )
        logger.info(f"resume best_metric: {best_metric}")
    else:
        resume = False
        abs_hydra_config_path = os.path.abspath(args.hydra_config_path)
        initialize_config_dir(
            config_dir=os.path.dirname(abs_hydra_config_path), version_base=None
        )
        cfg = compose(config_name=os.path.basename(abs_hydra_config_path))
        best_metric = -1

    device = cfg.device
    if not resume:
        model = hydra.utils.instantiate(cfg.model.obj).to(device)
        if cfg.model.get("pretrained_weight") is not None:
            model.load_from(torch.load(cfg.model.pretrained_weight))
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
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
    lr_scheduler = LrScheduleHandler(
        hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    )
    metrics = hydra.utils.instantiate(cfg.metrics)

    trainer_kwargs = dict(
        prepare_batch=prepare_batch,
    )
    trainer_handlers = [lr_scheduler]
    trainer = create_supervised_trainer(
        model, optimizer, loss_fn, device, **trainer_kwargs
    )
    for handler in trainer_handlers:
        handler.attach(trainer)

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

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.val_check_interval))
    def trainer_epoch_completed_val(engine):
        evaluator.run(val_dataloader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def trainer_epoch_completed(engine):
        mlflow.log_metric("epoch", value=engine.state.epoch, step=engine.state.epoch)

    if resume:
        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)
            engine.state.epoch = resume_epoch

    evaluator_kwargs = dict(
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (
            [eval_post_pred_transforms(i) for i in y_pred],
            [eval_post_label_transforms(i) for i in y],
        ),
        model_fn=lambda model, x: inferer(x, model),
    )
    evaluator = create_supervised_evaluator(model, metrics, device, **evaluator_kwargs)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def evaluator_epoch_completed(engine):
        nonlocal best_metric
        for name, value in engine.state.metrics.items():
            if name == cfg.key_metric_name:
                if value > best_metric:
                    best_metric = value
                    with tempfile.TemporaryDirectory() as tempdir:
                        checkpoint = {
                            "optimizer": optimizer,
                            "model": model,
                            "epoch": trainer.state.epoch,
                        }
                        checkpoint_path = os.path.join(tempdir, "checkpoint.pth")
                        torch.save(checkpoint, checkpoint_path)
                        mlflow.log_artifact(checkpoint_path, artifact_path="model")
                    logger.info(
                        f"saved new best metric model [{trainer.state.epoch}]: {best_metric}"
                    )
            mlflow.log_metric(name, value=value, step=trainer.state.epoch)

    set_mlflow_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_id=args.resume_run_id) as run:
        if not resume:
            with tempfile.TemporaryDirectory() as tempdir:
                cfg_save_path = os.path.join(tempdir, "config.yaml")
                omegaconf.OmegaConf.save(cfg, cfg_save_path)
                mlflow.log_artifact(cfg_save_path, artifact_path="config")
        trainer.run(train_dataloader, max_epochs=cfg.max_epochs)


if __name__ == "__main__":
    train()
